import os
import argparse
import logging
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from medicai.models.pl_models import TimmLightningClassifier
from medicai.data.datasets import create_dataloaders, CachedDICOMDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path: str, model_name: str, num_classes: int) -> TimmLightningClassifier:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Name of the model architecture
        num_classes: Number of classes
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model {model_name} from {checkpoint_path}")
    model = TimmLightningClassifier(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=0.0  # Not used during inference
    )
    
    try:
        model.load(checkpoint_path)
        model.eval()
        return model
    except Exception as e:
        # If loading fails with our custom method, try direct TorchScript loading
        logger.warning(f"Failed to load model with custom loader: {e}")
        logger.info("Attempting direct TorchScript loading...")
        
        # For TorchScript models, we can directly load and return the model
        if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
            try:
                scripted_model = torch.jit.load(checkpoint_path, map_location=model.device)
                scripted_model.eval()
                logger.info("Successfully loaded TorchScript model")
                
                # Create a wrapper class to match our API
                class ModelWrapper:
                    def __init__(self, model):
                        self.model = model
                        self.device = next(model.parameters()).device
                    
                    def to(self, device):
                        self.model = self.model.to(device)
                        self.device = device
                        return self
                    
                    def eval(self):
                        self.model.eval()
                        return self
                    
                    def __call__(self, x):
                        return self.model(x)
                
                return ModelWrapper(scripted_model)
            except Exception as inner_e:
                logger.error(f"All loading attempts failed: {inner_e}")
                raise RuntimeError(f"Could not load model from {checkpoint_path}")
        else:
            raise e

def evaluate_model(model, test_loader, device="cuda"):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test data
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation results
    """
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("Running inference on test data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Return computed metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_confusion_matrix(cm, class_names, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    

def plot_roc_curve(probabilities, labels, class_names, output_path=None):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        probabilities: Predicted probabilities
        labels: True labels
        class_names: List of class names
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # One-vs-Rest ROC curve for each class
    for i, class_name in enumerate(class_names):
        # Prepare binary labels (current class vs the rest)
        binary_labels = (labels == i).astype(int)
        
        # Get probability for current class
        class_probs = probabilities[:, i]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve for this class
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {output_path}")
    

def plot_misclassified_examples(test_dataset, predictions, true_labels, class_names, indices=None, output_path=None):
    """
    Plot examples of misclassified images.
    
    Args:
        test_dataset: Dataset containing the images
        predictions: Predicted labels
        true_labels: True labels
        class_names: List of class names for labeling the plots
        indices: Indices of images to plot (default: random misclassified samples)
        output_path: Path to save the plot (optional)
    """
    # Find misclassified indices
    misclassified = np.where(predictions != true_labels)[0]
    
    if len(misclassified) == 0:
        logger.info("No misclassified examples found!")
        return
    
    # Select samples to display
    if indices is None:
        # Randomly select up to 9 misclassified samples
        indices = np.random.choice(
            misclassified, 
            size=min(9, len(misclassified)), 
            replace=False
        )
    
    # Create the plot
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(indices):
        if i >= 9:  # Maximum 9 images
            break
            
        # Get the image
        img, _ = test_dataset[idx]
        
        
        # Plot the image
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[predictions[idx]]}")
        plt.axis('off')
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved misclassified examples to {output_path}")

def plot_correctly_classified_examples(test_dataset, predictions, true_labels, class_names, indices=None, output_path=None):
    """
    Plot examples of correctly classified images.
    
    Args:
        test_dataset: Dataset containing the images
        predictions: Predicted labels
        true_labels: True labels
        class_names: List of class names for labeling the plots
        indices: Indices of images to plot (default: random correctly classified samples)
        output_path: Path to save the plot (optional)
    """
    # Find correctly classified indices
    correctly_classified = np.where(predictions == true_labels)[0]
    
    if len(correctly_classified) == 0:
        logger.info("No correctly classified examples found!")
        return
    
    # Select samples to display
    if indices is None:
        # Randomly select up to 9 correctly classified samples
        indices = np.random.choice(
            correctly_classified, 
            size=min(9, len(correctly_classified)), 
            replace=False
        )
    
    # Create the plot
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(indices):
        if i >= 9:  # Maximum 9 images
            break
            
        # Get the image
        img, _ = test_dataset[idx]
        
        # Plot the image
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Class: {class_names[true_labels[idx]]}\nCorrectly Predicted", fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved correctly classified examples to {output_path}")

def export_results_to_csv(metrics, output_path):
    """
    Export metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save the CSV file
    """
    # Create a DataFrame with general metrics
    results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1']
        ]
    }
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported results to {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained model performance')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='./split_dataset_latest/test', help='Directory with test data')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_224', help='Model architecture name')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--use_cache', action='store_true', help='Use cached DICOM files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test dataset from {args.test_dir}...")
    _, test_loader, class_info = create_dataloaders(
        train_path=args.test_dir,  # Using test_dir for both to get class info (only test_loader will be used)
        test_path=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        image_size=args.image_size
    )
    
    # Extract class information
    num_classes = len(class_info['class_map'])
    class_names = list(class_info['class_map'].keys())
    logger.info(f"Found {num_classes} classes: {class_names}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = load_trained_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_classes=num_classes
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Display results in terminal
    logger.info("===== Evaluation Results =====")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    # Generate confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    
    # Generate ROC curve
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(metrics['probabilities'], metrics['labels'], class_names, roc_path)
    
    # Export results to CSV
    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    export_results_to_csv(metrics, csv_path)
    
    # Get test dataset for visualization of misclassified examples
    test_dataset = CachedDICOMDataset(
        root_dir=args.test_dir,
        transform=None,  # Will handle transformations separately for visualization
        use_cache=args.use_cache
    )
    
    # Plot misclassified examples
    misclassified_path = os.path.join(args.output_dir, 'misclassified_examples.png')
    plot_misclassified_examples(
        test_dataset,
        metrics['predictions'],
        metrics['labels'],
        class_names,  # This is correct now as the function expects class_names
        output_path=misclassified_path
    )
    
    # Plot correctly classified examples
    correctly_classified_path = os.path.join(args.output_dir, 'correctly_classified_examples.png')
    plot_correctly_classified_examples(
        test_dataset,
        metrics['predictions'],
        metrics['labels'],
        class_names,
        output_path=correctly_classified_path
    )
    
    logger.info(f"All results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
