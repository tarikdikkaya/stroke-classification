# DICOM Inference Scripts

This directory contains scripts for performing inference on DICOM medical images using trained Vision Transformer (ViT) models.

## Files Overview

- `inference_single.py` - Predict a single DICOM image
- `batch_inference.py` - Batch prediction on multiple DICOM images
- `example_inference.py` - Examples showing how to use the inference scripts

## Prerequisites

Make sure you have:

1. A trained model checkpoint (`.ckpt`, `.pt`, or `.pth` file)
2. DICOM files to predict
3. Python environment with all required packages installed

## Single Image Inference

### Basic Usage

```bash
python inference_single.py --checkpoint outputs/your_model.ckpt --dicom_path path/to/image.dcm --num_classes 2
```

### With Class Names

```bash
python inference_single.py \
    --checkpoint outputs/your_model.ckpt \
    --dicom_path main_dataset/NormalKronik/100005.dcm \
    --train_dir split_dataset/train \
    --num_classes 2
```

### Full Options

```bash
python inference_single.py \
    --checkpoint outputs/your_model.ckpt \
    --dicom_path path/to/image.dcm \
    --model_name vit_small_patch16_224 \
    --num_classes 2 \
    --image_size 224 \
    --device cuda \
    --train_dir split_dataset/train
```

### Arguments

- `--checkpoint` (required): Path to model checkpoint file
- `--dicom_path` (required): Path to DICOM file to predict
- `--model_name`: Model architecture (default: vit_small_patch16_224)
- `--num_classes`: Number of classes in your dataset (default: 2)
- `--image_size`: Input image size (default: 224)
- `--device`: Device to use - cuda or cpu (default: auto-detect)
- `--train_dir`: Training directory to infer class names from folder structure
- `--class_map`: JSON file with class index to name mapping

## Batch Inference

### Basic Usage

```bash
python batch_inference.py --dicom_folder main_dataset/NormalKronik --checkpoint outputs/your_model.ckpt --num_classes 2
```

### With Custom Output

```bash
python batch_inference.py \
    --dicom_folder main_dataset/NormalKronik \
    --checkpoint outputs/your_model.ckpt \
    --train_dir split_dataset/train \
    --output_file my_predictions.csv \
    --confidence_threshold 0.8 \
    --num_classes 2
```

### Arguments

All arguments from single inference plus:

- `--dicom_folder` (required): Folder containing DICOM files
- `--output_file`: Output CSV file path (default: auto-generated with timestamp)
- `--confidence_threshold`: Minimum confidence for "high confidence" classification (default: 0.5)

## Output Format

### Single Inference Output

```
============================================================
PREDICTION RESULTS
============================================================
DICOM File: main_dataset/NormalKronik/100005.dcm
Predicted Class: NormalKronik (Index: 0)
Confidence: 0.8542 (85.42%)

All Class Probabilities:
------------------------------
  NormalKronik        : 0.8542 (85.42%)
  Other               : 0.1458 (14.58%)
============================================================
```

### Batch Inference Output

Creates a CSV file with columns:

- `filename`: DICOM filename
- `filepath`: Full path to DICOM file
- `predicted_class_idx`: Predicted class index
- `predicted_class_name`: Predicted class name
- `confidence`: Confidence score (0-1)
- `status`: HIGH_CONFIDENCE, LOW_CONFIDENCE, or ERROR
- `error`: Error message if prediction failed
- `prob_ClassName`: Individual probability for each class

Plus a summary:

```
================================================================================
BATCH INFERENCE SUMMARY
================================================================================
Total files processed: 150
Successful predictions: 147
Failed predictions: 3
Success rate: 98.00%
Results saved to: batch_inference_results_20250625_143022.csv

Confidence Distribution:
  High confidence (>=0.50): 142 (96.6%)
  Low confidence (<0.50): 5 (3.4%)

Predicted Class Distribution:
  NormalKronik: 89 (60.5%)
  Other: 58 (39.5%)
================================================================================
```

## Class Mapping

The scripts can automatically infer class names in several ways:

### 1. From Training Directory Structure

If you provide `--train_dir`, it will read the folder names:

```
split_dataset/train/
├── NormalKronik/
└── Other/
```

### 2. From JSON File

Create a JSON file mapping class indices to names:

```json
{
  "0": "NormalKronik",
  "1": "Other"
}
```

### 3. Default Numeric Labels

If no class mapping is provided, classes are labeled as `Class_0`, `Class_1`, etc.

## Model Checkpoint Formats

The scripts support multiple checkpoint formats:

1. **PyTorch Lightning checkpoints** (`.ckpt`): Full training state
2. **PyTorch model state** (`.pt`, `.pth`): Model weights only
3. **TorchScript models** (`.pt`): Compiled/scripted models

## Image Preprocessing

The inference scripts use the same preprocessing pipeline as training:

1. Load DICOM pixel array
2. Convert grayscale to RGB (3-channel)
3. Normalize to [0, 255] range
4. Resize to model input size (224x224 by default)
5. Apply ImageNet normalization
6. Convert to tensor

This ensures consistency between training and inference.

## Troubleshooting

### Common Issues

1. **"Input height doesn't match model"**: Make sure `--image_size` matches your model
2. **"No DICOM files found"**: Check folder path and file extensions
3. **"Failed to load model"**: Verify checkpoint path and format
4. **CUDA out of memory**: Use `--device cpu` or reduce batch processing

### Error Handling

- Failed DICOM files are logged but don't stop processing
- Batch inference continues even if individual files fail
- All errors are saved in the output CSV for later review

## Performance Tips

1. Use GPU (`--device cuda`) for faster inference
2. For batch processing, the model is loaded once and reused
3. DICOM files can be cached to speed up repeated processing
4. Use appropriate confidence thresholds to filter uncertain predictions

## Integration Examples

### Python Script Integration

```python
from inference_single import load_trained_model, preprocess_dicom, predict_single_image

# Load model
model = load_trained_model("outputs/model.ckpt", "vit_small_patch16_224", 2)

# Process image
image_tensor = preprocess_dicom("image.dcm", 224)
pred_idx, confidence, probs = predict_single_image(model, image_tensor, "cuda")

print(f"Prediction: Class {pred_idx} with {confidence:.3f} confidence")
```

### Automated Pipeline

```bash
#!/bin/bash
# Process all DICOM folders
for folder in main_dataset/*/; do
    echo "Processing $folder"
    python batch_inference.py \
        --dicom_folder "$folder" \
        --checkpoint outputs/best_model.ckpt \
        --output_file "results_$(basename $folder).csv" \
        --num_classes 2
done
```
