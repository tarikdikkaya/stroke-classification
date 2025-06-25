import os
import argparse
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm

from inference_single import load_trained_model, load_class_mapping, preprocess_dicom, predict_single_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def batch_inference(
    dicom_folder: str,
    checkpoint_path: str,
    model_name: str = 'vit_small_patch16_224',
    num_classes: int = 2,
    image_size: int = 224,
    device: str = 'cuda',
    train_dir: str = None,
    class_map_file: str = None,
    output_file: str = None,
    confidence_threshold: float = 0.5
):
    """
    Perform batch inference on multiple DICOM files.
    
    Args:
        dicom_folder: Folder containing DICOM files
        checkpoint_path: Path to trained model checkpoint
        model_name: Model architecture name
        num_classes: Number of classes
        image_size: Input image size
        device: Device to use (cuda/cpu)
        train_dir: Training directory to infer class names
        class_map_file: JSON file with class mapping
        output_file: Output CSV file path
        confidence_threshold: Minimum confidence threshold for predictions
    """
    
    # Find all DICOM files
    dicom_folder = Path(dicom_folder)
    dicom_files = list(dicom_folder.glob("*.dcm"))
    
    if not dicom_files:
        logger.error(f"No DICOM files found in {dicom_folder}")
        return
    
    logger.info(f"Found {len(dicom_files)} DICOM files for processing")
    
    # Load model
    logger.info("Loading trained model...")
    model = load_trained_model(checkpoint_path, model_name, num_classes, image_size)
    
    # Load class mapping
    class_map = load_class_mapping(train_dir, class_map_file)
    
    # Prepare output file
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"batch_inference_results_{timestamp}.csv"
    
    # Process files
    results = []
    successful_predictions = 0
    failed_predictions = 0
    
    logger.info(f"Starting batch inference on {len(dicom_files)} files...")
    
    with tqdm(dicom_files, desc="Processing DICOM files") as pbar:
        for dicom_file in pbar:
            try:
                # Preprocess image
                image_tensor = preprocess_dicom(str(dicom_file), image_size)
                
                # Make prediction
                predicted_class_idx, confidence, probabilities = predict_single_image(
                    model, image_tensor, device
                )
                
                # Get class name
                predicted_class_name = class_map.get(predicted_class_idx, f"Class_{predicted_class_idx}") if class_map else f"Class_{predicted_class_idx}"
                
                # Check confidence threshold
                prediction_status = "HIGH_CONFIDENCE" if confidence >= confidence_threshold else "LOW_CONFIDENCE"
                
                # Store results
                result = {
                    'filename': dicom_file.name,
                    'filepath': str(dicom_file),
                    'predicted_class_idx': predicted_class_idx,
                    'predicted_class_name': predicted_class_name,
                    'confidence': confidence,
                    'status': prediction_status,
                    'error': None
                }
                
                # Add individual class probabilities
                for i, prob in enumerate(probabilities):
                    class_name = class_map.get(i, f"Class_{i}") if class_map else f"Class_{i}"
                    result[f'prob_{class_name}'] = prob
                
                results.append(result)
                successful_predictions += 1
                
                pbar.set_postfix({
                    'Success': successful_predictions,
                    'Failed': failed_predictions,
                    'Last': f"{predicted_class_name} ({confidence:.3f})"
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {dicom_file.name}: {e}")
                result = {
                    'filename': dicom_file.name,
                    'filepath': str(dicom_file),
                    'predicted_class_idx': None,
                    'predicted_class_name': None,
                    'confidence': None,
                    'status': 'ERROR',
                    'error': str(e)
                }
                results.append(result)
                failed_predictions += 1
                
                pbar.set_postfix({
                    'Success': successful_predictions,
                    'Failed': failed_predictions,
                    'Last': 'ERROR'
                })
    
    # Save results to CSV
    if results:
        logger.info(f"Saving results to {output_file}")
        
        # Get all possible column names
        all_columns = set()
        for result in results:
            all_columns.update(result.keys())
        
        # Sort columns for consistent output
        base_columns = ['filename', 'filepath', 'predicted_class_idx', 'predicted_class_name', 
                       'confidence', 'status', 'error']
        prob_columns = sorted([col for col in all_columns if col.startswith('prob_')])
        column_order = base_columns + prob_columns
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_order)
            writer.writeheader()
            writer.writerows(results)
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH INFERENCE SUMMARY")
    print("="*80)
    print(f"Total files processed: {len(dicom_files)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Success rate: {successful_predictions/len(dicom_files)*100:.2f}%")
    print(f"Results saved to: {output_file}")
    
    if successful_predictions > 0:
        # Show confidence distribution
        confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        high_conf = sum(1 for c in confidences if c >= confidence_threshold)
        low_conf = len(confidences) - high_conf
        
        print(f"\nConfidence Distribution:")
        print(f"  High confidence (>={confidence_threshold:.2f}): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
        print(f"  Low confidence (<{confidence_threshold:.2f}): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
        
        # Show class distribution
        if class_map:
            class_counts = {}
            for result in results:
                if result['predicted_class_name']:
                    class_name = result['predicted_class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\nPredicted Class Distribution:")
            for class_name, count in sorted(class_counts.items()):
                percentage = count / successful_predictions * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch inference on multiple DICOM images')
    parser.add_argument('--dicom_folder', type=str, required=True, help='Folder containing DICOM files')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_224', help='Model architecture name')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--train_dir', type=str, help='Training directory to infer class names')
    parser.add_argument('--class_map', type=str, help='Path to JSON file containing class mapping')
    parser.add_argument('--output_file', type=str, help='Output CSV file path')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dicom_folder):
        logger.error(f"DICOM folder not found: {args.dicom_folder}")
        return
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    try:
        batch_inference(
            dicom_folder=args.dicom_folder,
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
            num_classes=args.num_classes,
            image_size=args.image_size,
            device=args.device,
            train_dir=args.train_dir,
            class_map_file=args.class_map,
            output_file=args.output_file,
            confidence_threshold=args.confidence_threshold
        )
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise


if __name__ == '__main__':
    main()
