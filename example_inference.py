#!/usr/bin/env python3
"""
Example script showing how to use the inference_single.py script.
This demonstrates different ways to run inference on DICOM images.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_inference_examples():
    """Run example inference commands."""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    print("="*80)
    print("DICOM INFERENCE EXAMPLES")
    print("="*80)
    
    # Example 1: Basic inference with minimal arguments
    print("\n1. Basic inference (assuming you have a trained model):")
    print("-" * 50)
    cmd1 = [
        sys.executable, "inference_single.py",
        "--checkpoint", "outputs/your_model_checkpoint.ckpt",  # Replace with your actual checkpoint
        "--dicom_path", "main_dataset/NormalKronik/100005.dcm",  # Example DICOM file
        "--num_classes", "2"  # Adjust based on your dataset
    ]
    print(" ".join(cmd1))
    print()
    
    # Example 2: Inference with class mapping from training directory
    print("2. Inference with class names from training directory:")
    print("-" * 50)
    cmd2 = [
        sys.executable, "inference_single.py",
        "--checkpoint", "outputs/your_model_checkpoint.ckpt",
        "--dicom_path", "main_dataset/NormalKronik/100005.dcm",
        "--train_dir", "split_dataset/train",  # To infer class names
        "--num_classes", "2"
    ]
    print(" ".join(cmd2))
    print()
    
    # Example 3: Inference with custom image size and GPU
    print("3. Inference with custom settings (GPU, different image size):")
    print("-" * 50)
    cmd3 = [
        sys.executable, "inference_single.py",
        "--checkpoint", "outputs/your_model_checkpoint.ckpt",
        "--dicom_path", "main_dataset/NormalKronik/100005.dcm",
        "--image_size", "224",  # Match your training image size
        "--device", "cuda",
        "--num_classes", "2"
    ]
    print(" ".join(cmd3))
    print()
    
    # Example 4: Batch processing multiple files
    print("4. Example Python code for batch processing:")
    print("-" * 50)
    batch_code = '''
import os
import subprocess
import sys
from pathlib import Path

def batch_inference(dicom_folder, checkpoint_path, output_file="results.txt"):
    """Process multiple DICOM files and save results."""
    
    dicom_files = list(Path(dicom_folder).glob("*.dcm"))
    results = []
    
    for dicom_file in dicom_files:
        cmd = [
            sys.executable, "inference_single.py",
            "--checkpoint", checkpoint_path,
            "--dicom_path", str(dicom_file),
            "--num_classes", "2"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                results.append(f"{dicom_file.name}: SUCCESS")
                print(f"✅ Processed: {dicom_file.name}")
            else:
                results.append(f"{dicom_file.name}: ERROR - {result.stderr}")
                print(f"❌ Failed: {dicom_file.name}")
        except Exception as e:
            results.append(f"{dicom_file.name}: EXCEPTION - {str(e)}")
            print(f"❌ Exception: {dicom_file.name} - {e}")
    
    # Save results
    with open(output_file, "w") as f:
        f.write("\\n".join(results))
    
    print(f"\\nResults saved to: {output_file}")

# Usage example:
# batch_inference("main_dataset/NormalKronik", "outputs/your_model.ckpt")
'''
    print(batch_code)
    
    print("\n" + "="*80)
    print("NOTES:")
    print("="*80)
    print("1. Replace 'your_model_checkpoint.ckpt' with your actual checkpoint file")
    print("2. Make sure the checkpoint file exists in the outputs directory")
    print("3. Adjust --num_classes based on your dataset (2 for binary classification)")
    print("4. Use --train_dir to automatically infer class names from folder structure")
    print("5. The script supports .ckpt (PyTorch Lightning), .pt, and .pth files")
    print("6. Images are automatically resized to match the model's expected input size")
    print("7. DICOM files are preprocessed the same way as during training")

if __name__ == "__main__":
    run_inference_examples()
