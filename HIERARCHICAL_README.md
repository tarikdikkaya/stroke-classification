# Hierarchical Stroke Classification System

An advanced two-stage hierarchical classification system for stroke detection using CT and MRI images.

## System Overview

### Stage 1: CT Binary Classifier

- **Classes**: `NormalKronik` vs `Other`
- **Purpose**: Initial screening and triage
- **Model**: EfficientNet-B4 (optimized for medical imaging)

### Stage 2: MRI Binary Classifier

- **Classes**: `Hiperakut` vs `Subakut`
- **Purpose**: Detailed stroke type classification
- **Model**: Swin Transformer (excellent for fine-grained features)

## Enhanced Algorithm Logic

```python
if CT_prediction == "NormalKronik" and CT_confidence >= 0.7:
    result = ["NormalKronik"]
    if any(MRI_class_confidence >= 0.6):
        result.append(f"Possible_{MRI_class}")
        flag_for_review = True

elif CT_prediction == "Other" or CT_confidence < 0.7:
    result = [primary_MRI_class]
    if secondary_MRI_confidence >= 0.5:
        result.append(f"Also_Consider_{secondary_MRI_class}")

if uncertainty > threshold:
    result.append("Requires_Manual_Review")
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install additional DICOM processing libraries
pip install gdcm pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg pylibjpeg-rle
```

## Usage

### 1. Prepare Datasets

```bash
# Create hierarchical dataset structure
python train_hierarchical.py --prepare_only --data_path "path/to/your/dataset"
```

### 2. Train Individual Models

#### Train CT Binary Classifier

```bash
python train.py \
    --model_name tf_efficientnet_b4_ns \
    --learning_rate 3e-5 \
    --max_epochs 40 \
    --batch_size 16 \
    --dataset_type ct_binary \
    --class_weights \
    --label_smoothing 0.1 \
    --train_dir ./ct_binary_dataset/train \
    --test_dir ./ct_binary_dataset/val \
    --output_dir ./models/ct_classifier
```

#### Train MRI Binary Classifier

```bash
python train.py \
    --model_name swin_small_patch4_window7_224 \
    --learning_rate 1e-4 \
    --max_epochs 35 \
    --batch_size 12 \
    --dataset_type mri_binary \
    --class_weights \
    --mixup_alpha 0.2 \
    --train_dir ./mri_binary_dataset/train \
    --test_dir ./mri_binary_dataset/val \
    --output_dir ./models/mri_classifier
```

### 3. Train Both Models Automatically

```bash
# Train both models with optimized configurations
python train_hierarchical.py --data_path "path/to/your/dataset" --output_dir ./models
```

### 4. Run Inference

#### Single Image Pair

```bash
python inference_hierarchical.py \
    --ct_model ./models/ct_classifier/best_model.pt \
    --mri_model ./models/mri_classifier/best_model.pt \
    --ct_image path/to/ct_image.dcm \
    --mri_image path/to/mri_image.dcm \
    --output results.json
```

#### Batch Inference

```bash
python inference_hierarchical.py \
    --ct_model ./models/ct_classifier/best_model.pt \
    --mri_model ./models/mri_classifier/best_model.pt \
    --ct_images_dir ./ct_images/ \
    --mri_images_dir ./mri_images/ \
    --batch \
    --output ./batch_results/
```

## Expected Performance

| Model            | Dataset      | Expected Accuracy | Training Time |
| ---------------- | ------------ | ----------------- | ------------- |
| EfficientNet-B4  | CT Binary    | 85-90%            | 2-3 hours     |
| Swin Transformer | MRI Binary   | 83-88%            | 3-4 hours     |
| Combined System  | Hierarchical | 87-92%            | -             |

## Key Commands to Get Started

```bash
# 1. Prepare your datasets
python train_hierarchical.py --prepare_only

# 2. Train both models
python train_hierarchical.py

# 3. Run inference
python inference_hierarchical.py --ct_model models/ct_classifier/best_model.pt --mri_model models/mri_classifier/best_model.pt --ct_image test_ct.dcm --mri_image test_mri.dcm
```
