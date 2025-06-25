"""DICOM dataset loading with caching capability."""
import os
import numpy as np
import torch
import pydicom
import hashlib
import pickle
import functools
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class CachedDICOMDataset(Dataset):
    """DICOM dataset with caching capability to speed up loading during training."""
    
    def __init__(
        self, 
        root_dir: str, 
        transform=None, 
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            transform: PyTorch transform to apply to the images
            use_cache: Whether to cache the processed DICOM files
            cache_dir: Directory to store cache files (defaults to .cache in root_dir)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_cache = use_cache
        self.samples = []
        self.class_map = {}
        
        # Setup cache directory
        self.cache_dir = cache_dir
        if self.use_cache and not self.cache_dir:
            self.cache_dir = os.path.join(Path(root_dir).parent, '.cache')
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Discover classes and build dataset
        self._build_dataset()
    
    def _build_dataset(self):
        """Discover classes and build the dataset."""
        # Automatic class discovery
        class_names = [d for d in os.listdir(self.root_dir) 
                      if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_map = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
        
        for class_name, label in self.class_map.items():
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_path):
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path) and file_name.lower().endswith((".dcm", ".jpg", ".png")):
                        try:
                            # Check if file is readable but don't load pixel data yet
                            if file_name.lower().endswith(".dcm"):
                                pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
                            else:
                                # For JPG/PNG files, just verify they can be opened
                                Image.open(file_path).verify()
                            self.samples.append((file_path, label))
                        except Exception as e:
                            logger.warning(f"⚠️ Error: {file_path} could not be read! Error: {e}")
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ Error: No usable files found in '{self.root_dir}'!")

        logger.info(f"✅ Loaded {len(self.samples)} files. Classes: {self.class_map}")
    
    def _get_cache_path(self, file_path: str) -> str:
        """Generate a unique cache filename for a file."""
        if not self.use_cache:
            return None
            
        # Create a unique filename based on the file path and last modification time
        file_stat = os.stat(file_path)
        hash_input = f"{file_path}_{file_stat.st_mtime}"
        filename_hash = hashlib.md5(hash_input.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{filename_hash}.pkl")
    
    @functools.lru_cache(maxsize=32)  # In-memory cache for most recently used images
    def _load_image_with_cache(self, file_path: str) -> np.ndarray:
        """Load an image file with caching for faster access."""
        cache_path = self._get_cache_path(file_path)
        
        # Try to load from cache first
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {file_path}: {e}")
        
        # Load image file based on its extension
        try:
            if file_path.lower().endswith(".dcm"):
                # Handle DICOM files
                dicom = pydicom.dcmread(file_path, force=True)
                image = dicom.pixel_array.astype(np.float32)
                
                # Handle grayscale images by converting to 3-channel
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                
                # Normalize to [0, 255]
                if np.max(image) > np.min(image):  # Avoid division by zero
                    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
                else:
                    image = np.zeros_like(image)
            else:
                # Handle JPG/PNG files
                pil_image = Image.open(file_path).convert("RGB")
                image = np.array(pil_image).astype(np.float32)
            
            image = image.astype(np.uint8)
            
            # Save to cache
            if self.use_cache and cache_path:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(image, f)
                except Exception as e:
                    logger.warning(f"Failed to write cache for {file_path}: {e}")
                    
            return image
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            # Return a placeholder black image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)
    def get_original_file_paths(self) -> List[str]:
        return [path for path, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        
        # Load image from cache or directly from file
        image = self._load_image_with_cache(file_path)
        
        # Convert to PIL Image for transformations
        image = transforms.ToPILImage()(image)
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
    def get_class_info(self) -> Dict:
        """Return class information."""
        return {
            "class_map": self.class_map,
            "class_counts": {name: sum(1 for _, lbl in self.samples if lbl == idx) 
                             for name, idx in self.class_map.items()}
        }

def create_dataloaders(
    train_path: str,
    test_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    use_cache: bool = True,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and test dataloaders.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        batch_size: Batch size for training and validation
        num_workers: Number of workers for data loading
        use_cache: Whether to use cached images
        image_size: Size of input images
        
    Returns:
        Training dataloader, test dataloader, and class information
    """
    # Define transformations for training (with enhanced augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # Resize larger first
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Random crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Medical images can be flipped vertically
        transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random translation and scaling
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),  # Blur augmentation
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Random erasing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CachedDICOMDataset(
        root_dir=train_path,
        transform=train_transform,
        use_cache=use_cache
    )
    
    test_dataset = CachedDICOMDataset(
        root_dir=test_path,
        transform=test_transform,
        use_cache=use_cache
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class information from training dataset
    class_info = train_dataset.get_class_info()
    
    logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(test_loader)} test batches")
    
    return train_loader, test_loader, class_info