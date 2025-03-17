#!/usr/bin/env python
"""
Comprehensive training script using the medicai package.
Supports multiple model architectures with optimized training.

Features:
- Uses cached DICOM loading for faster training
- Automatic mixed precision (FP16) training
- Gradient accumulation for larger effective batch sizes
- Model checkpointing and early stopping
- TensorBoard logging
- Comprehensive metrics and visualization
"""

import os
import sys
import logging
import torch
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import from our medicai package
from medicai.data.datasets import create_dataloaders
from medicai.models import EfficientNetV2L, SwinLarge, ViTB16, CvTW24

# =============================================
# Configuration Variables (edit these as needed)
# =============================================

# Dataset parameters
TRAIN_DATA = "split_dataset_latest/train"
VAL_DATA = "split_dataset_latest/test"
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 6
USE_CACHE = True

# Model parameters
MODEL = "efficientnetv2_l"  # Options: "efficientnetv2_l", "swin_large", "vit_b_16", "cvt_w24"
PRETRAINED = True
RESUME_FROM = None  # Path to checkpoint to resume from, None for new training

# Training parameters
EPOCHS = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
GRADIENT_ACCUMULATION_STEPS = 1
EARLY_STOPPING_PATIENCE = 10
USE_AMP = True  # Automatic mixed precision
BALANCE_CLASSES = False  # Use class weights for imbalanced datasets

# Output parameters
EXPERIMENT_NAME = None  # Will be auto-generated if None
OUTPUT_DIR = "output"

# =============================================
# Configure logging
# =============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment():
    """Setup experiment name and directories."""
    # Create experiment name if not provided
    global EXPERIMENT_NAME
    if EXPERIMENT_NAME is None:
        EXPERIMENT_NAME = f"{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment directory and paths
    experiment_path = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
    checkpoint_dir = os.path.join(experiment_path, "checkpoints")
    tensorboard_dir = os.path.join(experiment_path, "tensorboard")
    plots_dir = os.path.join(experiment_path, "plots")
    
    # Create directories
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "train_data": TRAIN_DATA,
        "val_data": VAL_DATA,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "use_cache": USE_CACHE,
        "model": MODEL,
        "pretrained": PRETRAINED,
        "resume_from": RESUME_FROM,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "use_amp": USE_AMP,
        "balance_classes": BALANCE_CLASSES,
        "experiment_name": EXPERIMENT_NAME,
        "output_dir": OUTPUT_DIR
    }
    
    with open(os.path.join(experiment_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
        
    return {
        "root": experiment_path,
        "checkpoints": checkpoint_dir,
        "tensorboard": tensorboard_dir,
        "plots": plots_dir
    }

def create_model(num_classes):
    """Create and initialize the model based on configuration."""
    logger.info(f"Creating model: {MODEL}")
    
    if MODEL == "efficientnetv2_l":
        model = EfficientNetV2L(
            num_classes=num_classes,
            pretrained=PRETRAINED,
            input_size=IMAGE_SIZE
        )
    elif MODEL == "swin_large":
        model = SwinLarge(
            num_classes=num_classes,
            pretrained=PRETRAINED,
            input_size=IMAGE_SIZE
        )
    elif MODEL == "vit_b_16":
        model = ViTB16(
            num_classes=num_classes,
            pretrained=PRETRAINED,
            input_size=IMAGE_SIZE
        )
    elif MODEL == "cvt_w24":
        model = CvTW24(
            num_classes=num_classes,
            pretrained=PRETRAINED,
            input_size=IMAGE_SIZE
        )
    else:
        raise ValueError(f"Unsupported model: {MODEL}")
    
    # Log model details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    return model

def plot_training_history(history, save_dir):
    """Plot and save training history."""
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_history.png"), dpi=300)
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_history.png"), dpi=300)
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_rate.png"), dpi=300)
    plt.close()

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and generate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return {
        "confusion_matrix": cm,
        "classification_report": report
    }

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_classification_report(report, save_path):
    """Save classification report to file."""
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a simplified text report
    text_report = []
    text_report.append("Classification Report:")
    text_report.append("=" * 50)
    text_report.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    text_report.append("-" * 50)
    
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            text_report.append(
                f"{class_name:<20} "
                f"{metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} "
                f"{metrics['f1-score']:<10.4f} "
                f"{metrics['support']:<10}"
            )
    
    text_report.append("-" * 50)
    text_report.append(f"Accuracy: {report['accuracy']:.4f}")
    text_report.append(f"Macro Avg: {report['macro avg']['f1-score']:.4f}")
    text_report.append(f"Weighted Avg: {report['weighted avg']['f1-score']:.4f}")
    
    # Write text report
    with open(save_path.replace('.json', '.txt'), 'w') as f:
        f.write("\n".join(text_report))

def main():
    """Main training function."""
    # Setup experiment directories
    dirs = setup_experiment()
    logger.info(f"Experiment name: {EXPERIMENT_NAME}")
    logger.info(f"Output directory: {dirs['root']}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=dirs['tensorboard'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info(f"Loading data from {TRAIN_DATA} and {VAL_DATA}")
    train_loader, val_loader, class_info = create_dataloaders(
        train_path=TRAIN_DATA,
        test_path=VAL_DATA,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_cache=USE_CACHE,
        image_size=IMAGE_SIZE
    )
    
    class_names = list(class_info["class_map"].keys())
    num_classes = len(class_names)
    class_counts = class_info["class_counts"]
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Class distribution: {class_counts}")
    
    # Calculate class weights if requested
    if BALANCE_CLASSES:
        counts = np.array(list(class_counts.values()))
        # Create class weights inversely proportional to count
        class_weights = torch.tensor((1.0 / counts) * (sum(counts) / len(counts)), dtype=torch.float32)
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Create model
    model = create_model(num_classes)
    
    # Log dataset and model info to TensorBoard
    writer.add_text("info/dataset", f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    writer.add_text("info/classes", f"Classes: {class_names}")
    writer.add_text("info/model", f"Model: {MODEL}, Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    
    training_results = model.train_model(
        train_loader=train_loader,
        test_loader=val_loader,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights,
        device=device,
        checkpoint_dir=dirs["checkpoints"],
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        use_amp=USE_AMP,
        grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,
        resume_from=RESUME_FROM
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Log training metrics to TensorBoard
    history = training_results["history"]
    for epoch in range(len(history["train_loss"])):
        writer.add_scalar("Loss/train", history["train_loss"][epoch], epoch)
        writer.add_scalar("Loss/val", history["val_loss"][epoch], epoch)
        writer.add_scalar("Accuracy/train", history["train_acc"][epoch], epoch)
        writer.add_scalar("Accuracy/val", history["val_acc"][epoch], epoch)
        writer.add_scalar("LR", history["learning_rates"][epoch], epoch)
    
    # Generate plots
    plot_training_history(history, dirs["plots"])
    
    # Final evaluation on validation set
    logger.info("Performing final evaluation...")
    eval_results = evaluate_model(model, val_loader, device, class_names)
    
    # Save confusion matrix
    plot_confusion_matrix(
        eval_results["confusion_matrix"], 
        class_names,
        os.path.join(dirs["plots"], "confusion_matrix.png")
    )
    
    # Save classification report
    save_classification_report(
        eval_results["classification_report"],
        os.path.join(dirs["root"], "classification_report.json")
    )
    
    # Save final results summary
    results_summary = {
        "best_accuracy": training_results["best_accuracy"],
        "best_epoch": training_results["best_epoch"],
        "training_time_seconds": training_time,
        "final_validation_accuracy": eval_results["classification_report"]["accuracy"],
        "class_f1_scores": {
            class_name: metrics["f1-score"]
            for class_name, metrics in eval_results["classification_report"].items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        },
        "macro_f1": eval_results["classification_report"]["macro avg"]["f1-score"],
        "weighted_f1": eval_results["classification_report"]["weighted avg"]["f1-score"]
    }
    
    with open(os.path.join(dirs["root"], "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final results
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {training_results['best_accuracy']:.2f}% at epoch {training_results['best_epoch']}")
    logger.info(f"Final validation accuracy: {eval_results['classification_report']['accuracy']*100:.2f}%")
    logger.info(f"Macro F1-score: {eval_results['classification_report']['macro avg']['f1-score']:.4f}")
    logger.info(f"Weighted F1-score: {eval_results['classification_report']['weighted avg']['f1-score']:.4f}")
    logger.info(f"Best model saved at: {training_results['best_model_path']}")

if __name__ == "__main__":
    main()