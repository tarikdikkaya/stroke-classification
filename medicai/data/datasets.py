"""DICOM dataset loading with caching capability."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from torchvision import transforms
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import functools
import hashlib
import logging

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
                    dicom_path = os.path.join(class_path, file_name)
                    if os.path.isfile(dicom_path) and file_name.endswith(".dcm"):
                        try:
                            # Check if dicom is readable but don't load pixel data yet
                            dicom = pydicom.dcmread(dicom_path, force=True, stop_before_pixels=True)
                            self.samples.append((dicom_path, label))
                        except Exception as e:
                            logger.warning(f"⚠️ Error: {dicom_path} could not be read! Error: {e}")
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ Error: No usable DICOM files found in '{self.root_dir}'!")

        logger.info(f"✅ Loaded {len(self.samples)} DICOM files. Classes: {self.class_map}")
    
    def _get_cache_path(self, dicom_path: str) -> str:
        """Generate a unique cache filename for a DICOM file."""
        if not self.use_cache:
            return None
            
        # Create a unique filename based on the file path and last modification time
        file_stat = os.stat(dicom_path)
        hash_input = f"{dicom_path}_{file_stat.st_mtime}"
        filename_hash = hashlib.md5(hash_input.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{filename_hash}.pkl")
    
    @functools.lru_cache(maxsize=32)  # In-memory cache for most recently used images
    def _load_dicom_with_cache(self, dicom_path: str) -> np.ndarray:
        """Load a DICOM file with caching for faster access."""
        cache_path = self._get_cache_path(dicom_path)
        
        # Try to load from cache first
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {dicom_path}: {e}")
        
        # Load DICOM file if not in cache
        dicom = pydicom.dcmread(dicom_path, force=True)
        image = dicom.pixel_array.astype(np.float32)
        
        # Handle grayscale images by converting to 3-channel
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize to [0, 255]
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)
        
        # Save to cache
        if self.use_cache and cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(image, f)
            except Exception as e:
                logger.warning(f"Failed to write cache for {dicom_path}: {e}")
                
        return image
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        dicom_path, label = self.samples[idx]
        
        # Load image from cache or directly from file
        image = self._load_dicom_with_cache(dicom_path)
        
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
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    image_size: int = 480
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and test dataloaders with transforms.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use caching for DICOM files
        image_size: Input image size for the model
        
    Returns:
        Tuple of (train_loader, test_loader, class_info)
    """
    # Data transformations for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data transformations for testing/inference (no augmentation)
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
        pin_memory=True,  # Speed up data transfer to GPU
        drop_last=True,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader, train_dataset.get_class_info()

# Function for inference data preparation
def prepare_single_dicom_for_inference(
    dicom_path: str,
    transform=None,
    image_size: int = 480
) -> torch.Tensor:
    """
    Prepare a single DICOM file for inference.
    
    Args:
        dicom_path: Path to the DICOM file
        transform: Optional transform to apply
        image_size: Input image size for the model
        
    Returns:
        Tensor ready for model inference
    """
    # Default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load DICOM file
    dicom = pydicom.dcmread(dicom_path, force=True)
    image = dicom.pixel_array.astype(np.float32)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize to [0, 255]
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    
    # Convert to PIL and apply transform
    image = transforms.ToPILImage()(image)
    image_tensor = transform(image)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]