#!/usr/bin/env python3
"""
Hierarchical Training Script
Train both CT and MRI models for the hierarchical stroke classification system.
"""

import os
import argparse
import logging
import subprocess
from pathlib import Path
from hierarchical_classifier import create_hierarchical_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_hierarchical_models(base_data_path: str, output_dir: str = "./models"):
    """
    Train both CT and MRI models for hierarchical classification
    """
    
    # Create output directories
    ct_output = Path(output_dir) / "ct_classifier"
    mri_output = Path(output_dir) / "mri_classifier"
    ct_output.mkdir(parents=True, exist_ok=True)
    mri_output.mkdir(parents=True, exist_ok=True)
    
    # Prepare datasets
    logger.info("Preparing hierarchical datasets...")
    ct_path, mri_path = create_hierarchical_datasets(base_data_path)
    
    # Training configurations
    ct_config = {
        'model_name': 'tf_efficientnet_b4_ns',
        'learning_rate': 3e-5,
        'max_epochs': 40,
        'batch_size': 16,
        'dataset_type': 'ct_binary'
    }
    
    mri_config = {
        'model_name': 'swin_small_patch4_window7_224',
        'learning_rate': 1e-4,
        'max_epochs': 35,
        'batch_size': 12,
        'dataset_type': 'mri_binary'
    }
    
    # Train CT Model (Binary: NormalKronik vs Other)
    logger.info("🚀 Training CT Binary Classifier...")
    ct_command = [
        'python', 'train.py',
        '--train_dir', str(ct_path / 'train'),
        '--test_dir', str(ct_path / 'val'),
        '--model_name', ct_config['model_name'],
        '--learning_rate', str(ct_config['learning_rate']),
        '--max_epochs', str(ct_config['max_epochs']),
        '--batch_size', str(ct_config['batch_size']),
        '--output_dir', str(ct_output),
        '--dataset_type', ct_config['dataset_type'],
        '--class_weights',  # Handle class imbalance
        '--label_smoothing', '0.1',  # Reduce overfitting
        '--image_size', '224'
    ]
    
    try:
        subprocess.run(ct_command, check=True)
        logger.info("✅ CT model training completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ CT model training failed: {e}")
        return False
    
    # Train MRI Model (Binary: Hiperakut vs Subakut)
    logger.info("🚀 Training MRI Binary Classifier...")
    mri_command = [
        'python', 'train.py',
        '--train_dir', str(mri_path / 'train'),
        '--test_dir', str(mri_path / 'val'),
        '--model_name', mri_config['model_name'],
        '--learning_rate', str(mri_config['learning_rate']),
        '--max_epochs', str(mri_config['max_epochs']),
        '--batch_size', str(mri_config['batch_size']),
        '--output_dir', str(mri_output),
        '--dataset_type', mri_config['dataset_type'],
        '--class_weights',  # Handle class imbalance
        '--mixup_alpha', '0.2',  # Data augmentation
        '--image_size', '224'
    ]
    
    try:
        subprocess.run(mri_command, check=True)
        logger.info("✅ MRI model training completed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ MRI model training failed: {e}")
        return False
    
    logger.info("🎉 Hierarchical model training completed!")
    logger.info(f"Models saved to:")
    logger.info(f"  CT Model: {ct_output}")
    logger.info(f"  MRI Model: {mri_output}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train hierarchical stroke classification models')
    parser.add_argument('--data_path', type=str, 
                       default='a:/Teknofest SYZ/ASAMA_2_DATASETS/Yarışma 2.aşama veri seti kümesi',
                       help='Base path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Output directory for trained models')
    parser.add_argument('--prepare_only', action='store_true',
                       help='Only prepare datasets without training')
    
    args = parser.parse_args()
    
    if args.prepare_only:
        logger.info("Preparing datasets only...")
        ct_path, mri_path = create_hierarchical_datasets(args.data_path)
        logger.info(f"Datasets prepared:")
        logger.info(f"  CT Dataset: {ct_path}")
        logger.info(f"  MRI Dataset: {mri_path}")
        logger.info("Dataset preparation completed!")
        return
    
    # Train models
    success = train_hierarchical_models(args.data_path, args.output_dir)
    
    if success:
        logger.info("✅ All models trained successfully!")
        
        # Print usage instructions
        print("\n" + "="*60)
        print("HIERARCHICAL CLASSIFICATION SYSTEM READY!")
        print("="*60)
        print(f"CT Model: {args.output_dir}/ct_classifier/")
        print(f"MRI Model: {args.output_dir}/mri_classifier/")
        print("\nTo use the hierarchical classifier:")
        print("```python")
        print("from hierarchical_classifier import HierarchicalStrokeClassifier")
        print("")
        print("classifier = HierarchicalStrokeClassifier(")
        print(f"    ct_model_path='{args.output_dir}/ct_classifier/best_model.pt',")
        print(f"    mri_model_path='{args.output_dir}/mri_classifier/best_model.pt'")
        print(")")
        print("")
        print("result = classifier.classify(ct_image, mri_image)")
        print("print(result.final_prediction)")
        print("```")
        print("="*60)
    else:
        logger.error("❌ Training failed!")

if __name__ == '__main__':
    main()
