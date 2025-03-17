import timm
import torch
import os
import numpy as np
from pathlib import Path
import pydicom
from torchvision import transforms
from typing import Dict, Optional, Union, List
from medicai.models.vit import ViTB16  # Import the ViTB16 model class
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm

# Define the checkpoint path
checkpoint_path = "output/vit_b_16_20250316_071720/checkpoints/logs/lightning_logs/version_0/checkpoints/output/vit_b_16_20250316_071720/checkpoints/best_model-epoch=31-val_loss=0.0120.ckpt"

# Load the model using PyTorch Lightning's load_from_checkpoint
model = ViTB16.load_from_checkpoint(
    checkpoint_path,
    num_classes=2,
    strict=False  # Use strict=False if some keys in the checkpoint don't match the model
)
# Set model to evaluation mode
model.eval()

def infer_and_get_results(
    dicom_path: str,
    model: torch.nn.Module,
    class_names: Dict[int, str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Process a DICOM file and perform inference with the provided model.
    
    Args:
        dicom_path: Path to the DICOM file
        model: PyTorch model for inference
        class_names: Dictionary mapping class indices to class names
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Dictionary with prediction results
    """
    # ViT model uses 384x384 input size based on the model configuration
    image_size = 384
    
    # Define transform to match the model's expected input
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
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    
    # Move to the appropriate device
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction
        _, predicted_class = torch.max(probabilities, 1)
        confidence = probabilities[0][predicted_class].item()
        predicted_idx = predicted_class.item()
    
    # Prepare results dictionary
    results = {
        "predicted_class": predicted_idx,
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy(),
        "file": dicom_path
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Define class names
    class_names = {0: "Normal", 1: "Abnormal"}
    
    # Path to test dataset
    test_dir = "split_dataset_latest/test"
    
    # Lists to store predictions and ground truth
    all_preds = []
    all_true = []
    all_results = []
    
    # Get all DICOM files in the test directory (and subdirectories)
    dicom_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    print(f"Found {len(dicom_files)} DICOM files in {test_dir}")
    
    # Process each file with tqdm progress bar
    for dicom_path in tqdm(dicom_files, desc="Processing DICOM files"):
        # Determine ground truth from directory structure
        # Assuming directory structure like: split_dataset_latest/test/class_0/ for class 0
        parts = dicom_path.split(os.path.sep)
        test_index = parts.index('test')
        if test_index + 1 < len(parts):
            try:
                # Extract class number from folder name (class_0 or class_1)
                class_folder = parts[test_index + 1]
                if class_folder.startswith('class_'):
                    true_class = int(class_folder.split('_')[1])
                    # Perform inference
                    result = infer_and_get_results(dicom_path, model, class_names)
                    
                    # Store results
                    pred_class = result["predicted_class"]
                    all_preds.append(pred_class)
                    all_true.append(true_class)
                    all_results.append(result)
                else:
                    continue
            except (ValueError, IndexError):
                continue
        else:
            continue
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average='weighted')
    recall = recall_score(all_true, all_preds, average='weighted')
    f1 = f1_score(all_true, all_preds, average='weighted')
    
    # Print statistics
    print("\n===== Validation Statistics =====")
    print(f"Total samples: {len(all_preds)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_names[i] for i in range(len(class_names))],
                yticklabels=[class_names[i] for i in range(len(class_names))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix
    output_dir = "output/validation_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")


