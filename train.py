import os
import argparse
import logging
from datetime import datetime
import pytorch_lightning as pl

from medicai.models.pl_models import TimmLightningClassifier
from medicai.data.datasets import create_dataloaders
from model_configs import display_model_menu, get_model_config, print_model_recommendations

import torch
torch.set_float32_matmul_precision('high')

# Enable optimized attention for better FP16 performance if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a ViT model on medical images')
    
    parser.add_argument('--train_dir', type=str, default='/home/sezer/split_dataset_latest/train', help='Directory with training data')
    parser.add_argument('--test_dir', type=str, default='/home/sezer/split_dataset_latest/test', help='Directory with test/validation data')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-detected if not specified)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate (auto-detected if not specified)')
    parser.add_argument('--image_size', type=int, default=None, help='Input image size (auto-detected if not specified)')
    parser.add_argument('--use_cache', action='store_true', help='Use cached DICOM files')
    parser.add_argument('--use_fp16', action='store_true', default=True, help='Use FP16 mixed precision training')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of epochs to train')
    parser.add_argument('--model_id', type=int, default=None, help='Model ID from menu (1-22)')
    parser.add_argument('--auto_config', action='store_true', help='Use automatic configuration without menu')
    parser.add_argument('--dataset_type', type=str, default='general', 
                       choices=['general', 'ct_binary', 'mri_binary'], 
                       help='Type of dataset for hierarchical training')
    parser.add_argument('--class_weights', action='store_true', 
                       help='Use class weights to handle imbalanced datasets')
    parser.add_argument('--label_smoothing', type=float, default=0.0, 
                       help='Label smoothing factor (0.0 to 0.2)')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, 
                       help='Mixup alpha parameter (0.0 to disable)')
    
    args = parser.parse_args()
    
    # Interactive model selection or use provided model_id
    if args.model_id is None and not args.auto_config:
        print("🚀 Welcome to Advanced Medical Image Classification Training!")
        print("Choose from our curated selection of top-performing models:")
        model_id = display_model_menu()
    elif args.model_id is not None:
        model_id = args.model_id
    else:
        # Default for auto config
        model_id = 1  # tf_efficientnet_b4_ns
    
    # Get optimized configuration
    model_config = get_model_config(model_id, args.use_fp16)
    
    # Override with command line arguments if provided
    if args.batch_size is not None:
        model_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        model_config['learning_rate'] = args.learning_rate
    if args.image_size is not None:
        model_config['input_size'] = args.image_size
    
    # Print optimized settings
    if not args.auto_config:
        print_model_recommendations(model_config, args.use_fp16)
        
        # Confirm settings
        confirm = input("\n🤔 Proceed with these settings? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes', '']:
            print("❌ Training cancelled.")
            return
    
    # Extract settings
    model_name = model_config['name']
    batch_size = model_config['batch_size']
    learning_rate = model_config['learning_rate'] 
    image_size = model_config['input_size']
    
    logger.info(f"🎯 Selected Model: {model_config['display_name']}")
    logger.info(f"📊 Settings: Batch={batch_size}, LR={learning_rate}, Size={image_size}px")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading datasets from {args.train_dir} and {args.test_dir}...")
    train_loader, val_loader, class_info = create_dataloaders(
        train_path=args.train_dir,
        test_path=args.test_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        image_size=image_size
    )
    
    num_classes = len(class_info['class_map'])
    logger.info(f"Found {num_classes} classes: {class_info['class_map']}")
    logger.info(f"Class counts: {class_info['class_counts']}")
    
    # Initialize model with enhanced configuration
    logger.info(f"Initializing {model_name} model for {args.dataset_type} dataset...")
    
    # Calculate class weights if requested
    class_weights = None
    if args.class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Extract labels for class weight calculation
        labels = []
        for _, label in train_loader.dataset.samples:
            labels.append(label)
        
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        class_weights = torch.FloatTensor(class_weights)
        logger.info(f"Using class weights: {class_weights}")
    
    model = TimmLightningClassifier(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        input_size=image_size,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha
    )
    
    # Create timestamp for model checkpoint naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Enhanced callbacks for better training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'p2_model_{model_name.replace(".", "_")}_{timestamp}_{{epoch:02d}}_{{val_acc:.4f}}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,  # Save top 3 models
        save_last=True  # Save the last model
    )
    
    # Early stopping to prevent overfitting
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=8,  # Stop if no improvement for 8 epochs
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # Train the model
    precision = "16-mixed" if args.use_fp16 else "32"
    logger.info(f"Starting training for {args.max_epochs} epochs with precision: {precision}...")
    
    # Create a safe logging directory
    safe_log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(safe_log_dir, exist_ok=True)
    
    trainer = model.train_model(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision=precision,
        logger_dir=safe_log_dir
    )
    
    # Export the model
    model_export_path = os.path.join(args.output_dir, f'{model_name.replace(".", "_")}_{timestamp}.pt')
    logger.info(f"Exporting model to {model_export_path}...")
    model.export(model_export_path)
    
    logger.info("Training complete!")
    logger.info(f"📁 Model saved to: {model_export_path}")
    logger.info(f"🎯 Best model settings: {model_config['display_name']}")
    logger.info(f"📊 Final settings used: Batch={batch_size}, LR={learning_rate}, Size={image_size}px")

if __name__ == '__main__':
    main()