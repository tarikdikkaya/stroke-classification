import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
import numpy as np
import pydicom
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from pathlib import Path

# Display GPU info at start
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Initial VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
else:
    print("CUDA not available, using CPU")

# Configuration
TRAIN_PATH = "split_dataset_latest/train"
TEST_PATH = "split_dataset_latest/test"
CACHE_DIR = Path("dicom_cache")
BATCH_SIZE = 16  # Start with 4, increase if stable
IMAGE_SIZE = 384  # Start with 224, then try 320, 384, etc.
MODEL_NAME = "tf_efficientnet_l2.ns_jft_in1k"  # Start with b0, then try b3, b5, etc.
LR = 1e-4
WEIGHT_DECAY = 5e-5
NUM_WORKERS = 6
PIN_MEMORY = True
USE_AMP = True  # Mixed precision training (16-bit)
USE_COMPILE = False  # Torch compile (experimental)

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Speed up training

# Ask user for stopping epoch
stop_epoch = 100

# Optimized DICOM dataset with caching
class CachedDICOMDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_dir=CACHE_DIR, preprocess=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_dir = cache_dir
        self.samples = []
        self.preprocess = preprocess
        
        # Create class mapping
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.class_map = {d.name: i for i, d in enumerate(sorted(class_dirs))}
        
        # Collect all samples
        for class_name, label in self.class_map.items():
            class_path = self.root_dir / class_name
            for file_path in class_path.glob("*.dcm"):
                self.samples.append((file_path, label))
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ Error: No DICOM files found in '{root_dir}'")
        
        print(f"✅ Found {len(self.samples)} DICOM files. Classes: {self.class_map}")
        
        # Preprocess and cache samples
        if self.preprocess:
            self._preprocess_and_cache()
    
    def _preprocess_and_cache(self):
        for idx, (dicom_path, label) in enumerate(tqdm(self.samples, desc="Preprocessing")):
            cache_path = self.cache_dir / f"{dicom_path.stem}_{label}.pt"
            
            if not cache_path.exists():
                try:
                    # Load DICOM
                    dicom = pydicom.dcmread(str(dicom_path), force=True)
                    image = dicom.pixel_array.astype(np.float32)
                    
                    # Handle grayscale images
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=2)  # Convert to RGB
                    
                    # Normalize to 0-255 range
                    min_val = np.min(image)
                    max_val = np.max(image)
                    if max_val > min_val:  # Avoid division by zero
                        image = ((image - min_val) / (max_val - min_val)) * 255.0
                    image = image.astype(np.uint8)
                    
                    # Convert to tensor and save
                    tensor = transforms.ToTensor()(image)
                    torch.save(tensor, cache_path)
                except Exception as e:
                    print(f"Error processing {dicom_path}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        dicom_path, label = self.samples[idx]
        cache_path = self.cache_dir / f"{dicom_path.stem}_{label}.pt"
        
        try:
            if cache_path.exists():
                # Load from cache
                image = torch.load(cache_path)
            else:
                # Fallback to loading original DICOM
                dicom = pydicom.dcmread(str(dicom_path), force=True)
                image = dicom.pixel_array.astype(np.float32)
                
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=2)
                
                # Normalize to 0-255
                min_val = np.min(image)
                max_val = np.max(image)
                if max_val > min_val:
                    image = ((image - min_val) / (max_val - min_val)) * 255.0
                image = image.astype(np.uint8)
                image = transforms.ToTensor()(image)
                
                # Save for next time
                if self.preprocess:
                    torch.save(image, cache_path)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading {dicom_path}: {e}")
            # Return a placeholder image in case of error
            empty_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return empty_image, label

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading train dataset...")
train_dataset = CachedDICOMDataset(root_dir=TRAIN_PATH, transform=train_transform)
print("Loading test dataset...")
test_dataset = CachedDICOMDataset(root_dir=TEST_PATH, transform=test_transform)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    drop_last=True,
    persistent_workers=(NUM_WORKERS > 0)
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0)
)

# Create model
print(f"Creating model: {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(train_dataset.class_map))
model.to(device)

# Compile model if requested
if USE_COMPILE and hasattr(torch, 'compile'):
    try:
        print("Compiling model...")
        model = torch.compile(model)
        print("Model compilation successful")
    except Exception as e:
        print(f"Model compilation failed: {e}")

# Set up loss, optimizer and scheduler
class_counts = [0] * len(train_dataset.class_map)
for _, label in train_dataset.samples:
    class_counts[label] += 1

# Calculate class weights
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Class weights: {class_weights.tolist()}")
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# For mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# For metrics tracking
class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

# For model checkpointing
best_acc = 0.0
best_model_path = f"best_{MODEL_NAME.replace('-', '_')}.pth"

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            # Collect for metrics
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Compute overall metrics
    val_loss = running_loss / len(val_loader.dataset)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    # Per-class metrics
    class_correct = [0] * len(train_dataset.class_map)
    class_total = [0] * len(train_dataset.class_map)
    
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(len(train_dataset.class_map)):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        class_name = [k for k, v in train_dataset.class_map.items() if v == i][0]
        print(f"  {class_name}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    return val_loss, accuracy

def train_epoch(model, train_loader, optimizer, criterion, epoch, scaler):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for i, (images, labels) in enumerate(progress):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward and optimize with gradient scaling
        optimizer.zero_grad()
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        progress.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'batch_time': f"{batch_time.avg:.3f}s", 
            'data_time': f"{data_time.avg:.3f}s",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    return losses.avg

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, scaler):
    global best_acc, best_model_path
    
    # Initial validation
    print("\nInitial validation...")
    val_loss, accuracy = validate(model, test_loader, criterion)
    print(f"Initial validation - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n🔵 Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, scaler)
        
        # Validate
        val_loss, accuracy = validate(model, test_loader, criterion)
        
        # Print epoch summary
        print(f"Epoch {epoch+1} summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save if it's the best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'class_map': train_dataset.class_map
            }, best_model_path)
            print(f"✅ New best model saved with accuracy: {best_acc:.4f}")
        
        # Check if we should stop
        if epoch + 1 >= stop_epoch:
            print(f"⏹ Stopping at epoch {epoch+1} as requested")
            break

# Main execution
try:
    # Ensure model can do a forward pass
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP):
        with torch.no_grad():
            _ = model(dummy_input)
    print("Forward pass successful!")
    
    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, stop_epoch, scaler)
    
    # Final message
    print(f"\n✅ Training complete! Best accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
except KeyboardInterrupt:
    print("\n⏹ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up and report final memory usage
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"Final VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
