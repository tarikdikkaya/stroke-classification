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
from model_configs import display_model_menu, get_model_config, print_model_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_hierarchical_models(base_data_path: str, output_dir: str = "./models", 
                             ct_model_id: int = None, mri_model_id: int = None, auto_config: bool = False):
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
    
    # Interactive model selection for CT and MRI
    if not auto_config:
        if ct_model_id is None:
            print("\n🩻 SELECT CT BINARY CLASSIFIER MODEL:")
            print("(Recommended: EfficientNet or ConvNeXt models for CT scans)")
            ct_model_id = display_model_menu()
        
        if mri_model_id is None:
            print("\n🧠 SELECT MRI BINARY CLASSIFIER MODEL:")
            print("(Recommended: Vision Transformers or Swin models for MRI scans)")
            mri_model_id = display_model_menu()
    else:
        # Default models for auto configuration
        ct_model_id = ct_model_id or 1   # tf_efficientnet_b4_ns
        mri_model_id = mri_model_id or 3  # swin_small_patch4_window7_224
    
    # Get optimized configurations
    ct_config = get_model_config(ct_model_id, use_fp16=True)
    mri_config = get_model_config(mri_model_id, use_fp16=True)
    
    # Override specific settings for medical imaging
    ct_config.update({
        'dataset_type': 'ct_binary',
        'label_smoothing': 0.1,  # Reduce overfitting for medical data
        'class_weights': True    # Handle class imbalance
    })
    
    mri_config.update({
        'dataset_type': 'mri_binary',
        'mixup_alpha': 0.2,     # Data augmentation for MRI
        'class_weights': True   # Handle class imbalance
    })
    
    # Print configurations
    if not auto_config:
        print("\n🩻 CT CLASSIFIER CONFIGURATION:")
        print_model_recommendations(ct_config, use_fp16=True)
        
        print("\n🧠 MRI CLASSIFIER CONFIGURATION:")
        print_model_recommendations(mri_config, use_fp16=True)
        
        confirm = input("\n🤔 Proceed with these settings for both models? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes', '']:
            print("❌ Training cancelled.")
            return False
    
    # Train CT Model (Binary: NormalKronik vs Other)
    logger.info("🚀 Training CT Binary Classifier...")
    ct_command = [
        'python', 'train.py',
        '--train_dir', str(ct_path / 'train'),
        '--test_dir', str(ct_path / 'val'),
        '--model_id', str(ct_model_id),
        '--auto_config',  # Skip interactive menu
        '--learning_rate', str(ct_config['learning_rate']),
        '--batch_size', str(ct_config['batch_size']),
        '--image_size', str(ct_config['input_size']),
        '--output_dir', str(ct_output),
        '--dataset_type', ct_config['dataset_type'],
        '--class_weights',  # Handle class imbalance
        '--label_smoothing', str(ct_config['label_smoothing']),
        '--max_epochs', '40'
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
        '--model_id', str(mri_model_id),
        '--auto_config',  # Skip interactive menu
        '--learning_rate', str(mri_config['learning_rate']),
        '--batch_size', str(mri_config['batch_size']),
        '--image_size', str(mri_config['input_size']),
        '--output_dir', str(mri_output),
        '--dataset_type', mri_config['dataset_type'],
        '--class_weights',  # Handle class imbalance
        '--mixup_alpha', str(mri_config['mixup_alpha']),
        '--max_epochs', '35'
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
    parser.add_argument('--ct_model_id', type=int, default=None,
                       help='CT model ID from menu (1-22)')
    parser.add_argument('--mri_model_id', type=int, default=None,
                       help='MRI model ID from menu (1-22)')
    parser.add_argument('--auto_config', action='store_true',
                       help='Use automatic configuration without interactive menu')
    
    args = parser.parse_args()
    
    if args.prepare_only:
        logger.info("Preparing datasets only...")
        ct_path, mri_path = create_hierarchical_datasets(args.data_path)
        logger.info(f"Datasets prepared:")
        logger.info(f"  CT Dataset: {ct_path}")
        logger.info(f"  MRI Dataset: {mri_path}")
        logger.info("Dataset preparation completed!")
        return
    
    # Welcome message
    if not args.auto_config:
        print("🏥 Welcome to Hierarchical Medical Image Classification Training!")
        print("This system trains separate models for CT and MRI classification.")
        print("CT Model: Classifies Normal/Chronic vs Other conditions")
        print("MRI Model: Classifies Hyperacute vs Subacute stroke")
        print()
    
    # Train models
    success = train_hierarchical_models(
        args.data_path, 
        args.output_dir,
        args.ct_model_id,
        args.mri_model_id,
        args.auto_config
    )
    
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
