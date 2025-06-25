"""
Test script to verify FP16 training functionality.
"""
import torch
import sys
import os

def check_fp16_support():
    """Check if FP16 training is supported on the current system."""
    print("🔍 Checking FP16 training support...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available - FP16 training requires GPU support")
        return False
    
    # Check GPU compute capability for FP16
    device_count = torch.cuda.device_count()
    print(f"📱 Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {props.name}")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
        
        # Check if the GPU supports efficient FP16
        if props.major >= 7:  # Tensor cores available from compute capability 7.0
            print("    ✅ Supports efficient FP16 (Tensor Cores)")
        elif props.major >= 6:  # Basic FP16 support from compute capability 6.0
            print("    ⚠️  Supports basic FP16 (no Tensor Cores)")
        else:
            print("    ❌ Limited FP16 support")
    
    # Test basic FP16 operations
    try:
        device = torch.device('cuda')
        x = torch.randn(2, 3, 224, 224, dtype=torch.float16, device=device)
        y = torch.randn(2, 3, 224, 224, dtype=torch.float16, device=device)
        z = x + y
        print("✅ Basic FP16 operations work correctly")
        return True
    except Exception as e:
        print(f"❌ FP16 operations failed: {e}")
        return False

def test_model_precision():
    """Test model creation and precision settings."""
    print("\n🧪 Testing model precision configuration...")
    
    try:
        # Add the medicai module to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from medicai.models.pl_models import TimmLightningClassifier
        
        # Create a test model
        model = TimmLightningClassifier(
            model_name='vit_small_patch16_224',
            num_classes=3,
            learning_rate=1e-4,
            input_size=224
        )
        
        print("✅ Model created successfully")
        
        # Test precision configuration
        model.configure_model_for_precision("16-mixed")
        print("✅ FP16 precision configuration applied")
        
        # Test with dummy data
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = model.to(device)
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            
            # Test forward pass
            with torch.no_grad():
                output = model(dummy_input)
                print(f"✅ Forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing FP16 Training Setup")
    print("=" * 50)
    
    fp16_supported = check_fp16_support()
    model_works = test_model_precision()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"  FP16 Support: {'✅ PASS' if fp16_supported else '❌ FAIL'}")
    print(f"  Model Tests:  {'✅ PASS' if model_works else '❌ FAIL'}")
    
    if fp16_supported and model_works:
        print("\n🎉 Your system is ready for FP16 training!")
        print("\nTo use FP16 training, run:")
        print("  python train.py --use_fp16")
        print("  or")
        print("  python train.py --use_fp16 --batch_size 16")
    else:
        print("\n⚠️  There may be issues with FP16 training on this system.")

if __name__ == "__main__":
    main()
