import os
import argparse
import logging
from datetime import datetime
import pytorch_lightning as pl

from medicai.models.pl_models import TimmLightningClassifier
from medicai.data.datasets import create_dataloaders

import torch
torch.set_float32_matmul_precision('high')

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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=448, help='Input image size')
    parser.add_argument('--use_cache', action='store_true', help='Use cached DICOM files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading datasets from {args.train_dir} and {args.test_dir}...")
    train_loader, val_loader, class_info = create_dataloaders(
        train_path=args.train_dir,
        test_path=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        image_size=args.image_size
    )
    
    num_classes = len(class_info['class_map'])
    logger.info(f"Found {num_classes} classes: {class_info['class_map']}")
    logger.info(f"Class counts: {class_info['class_counts']}")
    
    # Initialize model
    logger.info("Initializing ViT Small Patch16 model...")
    model = TimmLightningClassifier(
        model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        input_size=args.image_size
    )
    
    # Create timestamp for model checkpoint naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_{timestamp}_{{epoch:02d}}_{{val_acc:.4f}}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )
    
    # Train the model for 2 epochs
    logger.info("Starting training for 2 epochs...")
    trainer = model.train_model(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=30,  # Training for exactly 2 epochs
        callbacks=[checkpoint_callback]
    )
    
    # Export the model
    model_export_path = os.path.join(args.output_dir, f'vit_small_patch16_224_{timestamp}.pt')
    logger.info(f"Exporting model to {model_export_path}...")
    model.export(model_export_path)
    
    logger.info("Training complete!")

if __name__ == '__main__':
    main()
