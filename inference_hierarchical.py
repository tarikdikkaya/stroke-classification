#!/usr/bin/env python3
"""
Hierarchical Stroke Classification Inference Script
"""

import torch
import argparse
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from hierarchical_classifier import HierarchicalStrokeClassifier
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess a single image."""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Handle DICOM files
        if image_path.lower().endswith('.dcm'):
            import pydicom
            dicom = pydicom.dcmread(image_path)
            pixel_array = dicom.pixel_array
            
            # Convert to PIL Image
            if len(pixel_array.shape) == 2:
                # Grayscale
                pixel_array = ((pixel_array - pixel_array.min()) / 
                             (pixel_array.max() - pixel_array.min()) * 255).astype('uint8')
                image = Image.fromarray(pixel_array).convert('RGB')
            else:
                image = Image.fromarray(pixel_array)
        else:
            # Regular image files
            image = Image.open(image_path).convert('RGB')
        
        return transform(image)
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise

def run_inference(ct_model_path: str, mri_model_path: str, 
                 ct_image_path: str, mri_image_path: str,
                 output_file: str = None):
    """Run hierarchical inference on CT and MRI images."""
    
    # Initialize classifier
    classifier = HierarchicalStrokeClassifier(
        ct_model_path=ct_model_path,
        mri_model_path=mri_model_path,
        ct_threshold=0.7,
        mri_threshold=0.6,
        confidence_threshold=0.5
    )
    
    # Load and preprocess images
    logger.info("Loading images...")
    ct_image = load_and_preprocess_image(ct_image_path)
    mri_image = load_and_preprocess_image(mri_image_path)
    
    # Run classification
    logger.info("Running hierarchical classification...")
    result = classifier.classify(ct_image, mri_image)
    
    # Prepare output
    output_data = {
        'ct_image': ct_image_path,
        'mri_image': mri_image_path,
        'primary_class': result.primary_class,
        'confidence': result.confidence,
        'final_prediction': result.final_prediction,
        'secondary_classes': result.secondary_classes,
        'ct_uncertainty': result.ct_uncertainty,
        'mri_uncertainty': result.mri_uncertainty,
        'requires_review': result.requires_review
    }
    
    # Print results
    print("\n" + "="*60)
    print("HIERARCHICAL STROKE CLASSIFICATION RESULTS")
    print("="*60)
    print(f"CT Image: {ct_image_path}")
    print(f"MRI Image: {mri_image_path}")
    print(f"Primary Classification: {result.primary_class} (confidence: {result.confidence:.3f})")
    print(f"Final Prediction: {', '.join(result.final_prediction)}")
    
    if result.secondary_classes:
        print(f"Secondary Classes:")
        for class_name, conf in result.secondary_classes:
            print(f"  - {class_name}: {conf:.3f}")
    
    print(f"CT Uncertainty: {result.ct_uncertainty:.3f}")
    print(f"MRI Uncertainty: {result.mri_uncertainty:.3f}")
    print(f"Requires Manual Review: {'Yes' if result.requires_review else 'No'}")
    print("="*60)
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return result

def batch_inference(ct_model_path: str, mri_model_path: str,
                   ct_images_dir: str, mri_images_dir: str,
                   output_dir: str = "./inference_results"):
    """Run batch inference on directories of images."""
    
    ct_images_dir = Path(ct_images_dir)
    mri_images_dir = Path(mri_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize classifier
    classifier = HierarchicalStrokeClassifier(
        ct_model_path=ct_model_path,
        mri_model_path=mri_model_path
    )
    
    # Get matching image pairs
    ct_images = list(ct_images_dir.glob("*.dcm")) + list(ct_images_dir.glob("*.jpg")) + list(ct_images_dir.glob("*.png"))
    mri_images = list(mri_images_dir.glob("*.dcm")) + list(mri_images_dir.glob("*.jpg")) + list(mri_images_dir.glob("*.png"))
    
    results = []
    
    for i, (ct_path, mri_path) in enumerate(zip(ct_images, mri_images)):
        logger.info(f"Processing pair {i+1}/{len(ct_images)}: {ct_path.name}, {mri_path.name}")
        
        try:
            # Load images
            ct_image = load_and_preprocess_image(str(ct_path))
            mri_image = load_and_preprocess_image(str(mri_path))
            
            # Classify
            result = classifier.classify(ct_image, mri_image)
            
            # Store result
            result_data = {
                'ct_image': str(ct_path),
                'mri_image': str(mri_path),
                'primary_class': result.primary_class,
                'confidence': result.confidence,
                'final_prediction': result.final_prediction,
                'secondary_classes': result.secondary_classes,
                'ct_uncertainty': result.ct_uncertainty,
                'mri_uncertainty': result.mri_uncertainty,
                'requires_review': result.requires_review
            }
            results.append(result_data)
            
        except Exception as e:
            logger.error(f"Error processing {ct_path.name}, {mri_path.name}: {e}")
            continue
    
    # Save batch results
    output_file = output_dir / "batch_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch inference completed. Results saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Hierarchical stroke classification inference')
    parser.add_argument('--ct_model', type=str, required=True,
                       help='Path to trained CT model')
    parser.add_argument('--mri_model', type=str, required=True,
                       help='Path to trained MRI model')
    parser.add_argument('--ct_image', type=str,
                       help='Path to CT image for single inference')
    parser.add_argument('--mri_image', type=str,
                       help='Path to MRI image for single inference')
    parser.add_argument('--ct_images_dir', type=str,
                       help='Directory containing CT images for batch inference')
    parser.add_argument('--mri_images_dir', type=str,
                       help='Directory containing MRI images for batch inference')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch inference')
    
    args = parser.parse_args()
    
    if args.batch:
        if not (args.ct_images_dir and args.mri_images_dir):
            parser.error("Batch inference requires --ct_images_dir and --mri_images_dir")
        
        output_dir = args.output or "./inference_results"
        batch_inference(args.ct_model, args.mri_model, 
                       args.ct_images_dir, args.mri_images_dir, output_dir)
    else:
        if not (args.ct_image and args.mri_image):
            parser.error("Single inference requires --ct_image and --mri_image")
        
        run_inference(args.ct_model, args.mri_model,
                     args.ct_image, args.mri_image, args.output)

if __name__ == '__main__':
    main()
