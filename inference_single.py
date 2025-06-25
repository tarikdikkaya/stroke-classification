import os
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pydicom
import json
from torchvision import transforms
from medicai.models.pl_models import TimmLightningClassifier
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path: str, model_name: str, num_classes: int, input_size: int = 224) -> TimmLightningClassifier:
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model {model_name} from {checkpoint_path}")
    
    # Create model instance
    model = TimmLightningClassifier(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=1e-4,  # Not used during inference
        input_size=input_size
    )
    
    try:
        # Try loading PyTorch Lightning checkpoint
        if checkpoint_path.endswith('.ckpt'):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Successfully loaded PyTorch Lightning checkpoint")
        elif checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
            # Try loading as regular PyTorch model
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded PyTorch model")
            except:
                # Try loading as TorchScript
                scripted_model = torch.jit.load(checkpoint_path, map_location='cpu')
                logger.info("Successfully loaded TorchScript model")
                return scripted_model
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}")
        raise RuntimeError(f"Could not load model from {checkpoint_path}")


def load_class_mapping(train_dir: str = None, class_map_file: str = None) -> dict:
    """Load class mapping from training directory or class map file."""
    class_map = {}
    
    if class_map_file and os.path.exists(class_map_file):
        # Load from JSON file
        try:
            with open(class_map_file, 'r') as f:
                class_map = json.load(f)
            logger.info(f"Loaded class mapping from {class_map_file}: {class_map}")
        except Exception as e:
            logger.warning(f"Failed to load class mapping from file: {e}")
    
    elif train_dir and os.path.exists(train_dir):
        # Infer from training directory structure
        class_names = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
        class_map = {idx: class_name for idx, class_name in enumerate(sorted(class_names))}
        logger.info(f"Inferred class mapping from {train_dir}: {class_map}")
    
    else:
        # Default fallback
        logger.warning("No class mapping provided, using default numeric labels")
        class_map = {}
    
    return class_map

def preprocess_dicom(dicom_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Preprocess DICOM file to match training pipeline preprocessing.
    This should match the preprocessing in medicai.data.datasets.
    """
    try:
        # Load DICOM file
        dicom = pydicom.dcmread(dicom_path, force=True)
        image = dicom.pixel_array.astype(np.float32)
        
        # Handle grayscale images by converting to 3-channel (RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize to [0, 255] range
        if np.max(image) > np.min(image):  # Avoid division by zero
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        else:
            image = np.zeros_like(image)
        
        # Convert to uint8 and then to PIL Image
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Apply the same transforms as during training (without augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms and add batch dimension
        tensor = transform(pil_image).unsqueeze(0)
        
        logger.info(f"Successfully preprocessed DICOM: {dicom_path} -> shape: {tensor.shape}")
        return tensor
        
    except Exception as e:
        logger.error(f"Failed to preprocess DICOM {dicom_path}: {e}")
        raise


def predict_single_image(model, image_tensor: torch.Tensor, device: str = 'cpu') -> tuple:
    """
    Predict class and confidence for a single image.
    
    Returns:
        tuple: (predicted_class_idx, confidence_scores, probabilities)
    """
    # Move model and data to device
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass
        logits = model(image_tensor)
        
        # Get probabilities using softmax
        probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class and confidence
        confidence_scores, predicted_class = torch.max(probabilities, 1)
        
        predicted_class_idx = predicted_class.item()
        confidence = confidence_scores.item()
        probs = probabilities.squeeze().cpu().numpy()
        
    return predicted_class_idx, confidence, probs

def main():
    parser = argparse.ArgumentParser(description='Classify a single DICOM image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.ckpt, .pt, .pth)')
    parser.add_argument('--dicom_path', type=str, required=True, help='Path to a single DICOM file')
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_224', help='Model architecture name')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--train_dir', type=str, help='Training directory to infer class names')
    parser.add_argument('--class_map', type=str, help='Path to JSON file containing class mapping')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.dicom_path):
        logger.error(f"DICOM file not found: {args.dicom_path}")
        return
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    try:
        # Load the trained model
        logger.info("Loading trained model...")
        model = load_trained_model(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
            num_classes=args.num_classes,
            input_size=args.image_size
        )
        
        # Load class mapping
        class_map = load_class_mapping(args.train_dir, args.class_map)
        
        # Preprocess the DICOM image
        logger.info(f"Preprocessing DICOM image: {args.dicom_path}")
        image_tensor = preprocess_dicom(args.dicom_path, args.image_size)
        
        # Make prediction
        logger.info("Making prediction...")
        predicted_class_idx, confidence, probabilities = predict_single_image(
            model, image_tensor, args.device
        )
        
        # Format results
        if class_map and predicted_class_idx in class_map:
            predicted_class_name = class_map[predicted_class_idx]
        else:
            predicted_class_name = f"Class_{predicted_class_idx}"
        
        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"DICOM File: {args.dicom_path}")
        print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_idx})")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print("\nAll Class Probabilities:")
        print("-" * 30)
        
        for i, prob in enumerate(probabilities):
            class_name = class_map.get(i, f"Class_{i}") if class_map else f"Class_{i}"
            print(f"  {class_name:<20}: {prob:.4f} ({prob*100:.2f}%)")
        
        print("="*60)
        
        # Log the result
        logger.info(f"Prediction complete: {predicted_class_name} with {confidence:.4f} confidence")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == '__main__':
    main()
