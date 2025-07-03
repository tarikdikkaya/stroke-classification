"""
Model configurations for TimM top-performing models with optimized settings.
Based on Hugging Face TimM top 20 ImageNet-1K models collection.
"""

from typing import Dict, Any

# Top TimM models with optimized configurations
MODEL_CONFIGS = {
    # Current default models (maintained for compatibility)
    1: {
        'name': 'tf_efficientnet_b4_ns',
        'display_name': 'EfficientNet-B4 (Noisy Student)',
        'input_size': 380,
        'batch_size': 8,
        'learning_rate': 3e-5,
        'description': 'Current default - Good balance of speed and accuracy'
    },
    2: {
        'name': 'vit_small_patch16_224',
        'display_name': 'Vision Transformer Small',
        'input_size': 224,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'description': 'Lightweight ViT model'
    },
    3: {
        'name': 'swin_small_patch4_window7_224',
        'display_name': 'Swin Transformer Small',
        'input_size': 224,
        'batch_size': 12,
        'learning_rate': 1e-4,
        'description': 'Current MRI default - Hierarchical attention'
    },
    
    # Top TimM models from Hugging Face collection
    4: {
        'name': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        'display_name': 'EVA-02 Large (448px) - SOTA Performance',
        'input_size': 448,
        'batch_size': 4,
        'learning_rate': 5e-6,
        'description': '🏆 Top accuracy - Requires 24GB+ VRAM'
    },
    5: {
        'name': 'eva02_base_patch14_448.mim_in22k_ft_in1k',
        'display_name': 'EVA-02 Base (448px) - High Performance',
        'input_size': 448,
        'batch_size': 6,
        'learning_rate': 1e-5,
        'description': '🥈 Excellent accuracy - Requires 16GB+ VRAM'
    },
    6: {
        'name': 'convnextv2_huge.fcmae_ft_in22k_in1k_512',
        'display_name': 'ConvNeXt-V2 Huge (512px) - Ultimate Performance',
        'input_size': 512,
        'batch_size': 2,
        'learning_rate': 3e-6,
        'description': '🚀 Cutting-edge CNN - Requires 32GB+ VRAM'
    },
    7: {
        'name': 'beit_large_patch16_512.in22k_ft_in22k_in1k',
        'display_name': 'BEiT Large (512px) - Self-supervised Excellence',
        'input_size': 512,
        'batch_size': 3,
        'learning_rate': 5e-6,
        'description': '🎯 Pre-trained with masked modeling'
    },
    8: {
        'name': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
        'display_name': 'ConvNeXt-V2 Large (384px) - Modern CNN',
        'input_size': 384,
        'batch_size': 6,
        'learning_rate': 5e-6,
        'description': '💪 Advanced ConvNet architecture'
    },
    9: {
        'name': 'eva_large_patch14_336.in22k_ft_in1k',
        'display_name': 'EVA Large (336px) - Excellent Balance',
        'input_size': 336,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'description': '⚡ Great performance/efficiency ratio'
    },
    10: {
        'name': 'beit_large_patch16_384.in22k_ft_in22k_in1k',
        'display_name': 'BEiT Large (384px) - Self-supervised',
        'input_size': 384,
        'batch_size': 6,
        'learning_rate': 5e-6,
        'description': '🧠 Advanced self-supervised learning'
    },
    11: {
        'name': 'convnext_xlarge_384_in22ft1k',
        'display_name': 'ConvNeXt XLarge (384px) - CNN Power',
        'input_size': 384,
        'batch_size': 4,
        'learning_rate': 3e-6,
        'description': '🔥 Powerful CNN for medical imaging'
    },
    12: {
        'name': 'tf_efficientnetv2_l.in21k_ft_in1k',
        'display_name': 'EfficientNet-V2 Large - Optimized Training',
        'input_size': 480,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'description': '⚡ Fast training with progressive resizing'
    },
    13: {
        'name': 'maxvit_large_tf_512.in21k_ft_in1k',
        'display_name': 'MaxViT Large (512px) - Hybrid Architecture',
        'input_size': 512,
        'batch_size': 3,
        'learning_rate': 5e-6,
        'description': '🌟 CNN + Transformer hybrid'
    },
    14: {
        'name': 'convnextv2_base.fcmae_ft_in22k_in1k_384',
        'display_name': 'ConvNeXt-V2 Base (384px) - Balanced',
        'input_size': 384,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'description': '💯 Good balance for medical data'
    },
    15: {
        'name': 'eva_large_patch14_196.in22k_ft_in1k',
        'display_name': 'EVA Large (196px) - Efficient',
        'input_size': 224,  # Resized for efficiency
        'batch_size': 12,
        'learning_rate': 1e-5,
        'description': '🏃 Fast training, good accuracy'
    },
    16: {
        'name': 'beitv2_large_patch16_224.in1k_ft_in22k_in1k',
        'display_name': 'BEiT-v2 Large (224px) - Updated',
        'input_size': 224,
        'batch_size': 12,
        'learning_rate': 5e-6,
        'description': '🆕 Improved BEiT architecture'
    },
    17: {
        'name': 'convnext_large_384_in22ft1k',
        'display_name': 'ConvNeXt Large (384px) - Proven',
        'input_size': 384,
        'batch_size': 6,
        'learning_rate': 5e-6,
        'description': '🎯 Reliable choice for medical imaging'
    },
    18: {
        'name': 'tf_efficientnet_b7.ns_jft_in1k',
        'display_name': 'EfficientNet-B7 (Noisy Student) - Classic',
        'input_size': 600,
        'batch_size': 2,
        'learning_rate': 1e-5,
        'description': '🏛️ Classic high-performance model'
    },
    19: {
        'name': 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
        'display_name': 'Swin-V2 Large (384px) - Advanced Attention',
        'input_size': 384,
        'batch_size': 6,
        'learning_rate': 5e-6,
        'description': '🔄 Advanced window attention'
    },
    20: {
        'name': 'maxvit_base_tf_512.in21k_ft_in1k',
        'display_name': 'MaxViT Base (512px) - Efficient Hybrid',
        'input_size': 512,
        'batch_size': 4,
        'learning_rate': 1e-5,
        'description': '⚖️ Balanced hybrid architecture'
    },
    21: {
        'name': 'convnext_base_384_in22ft1k',
        'display_name': 'ConvNeXt Base (384px) - Medical Optimized',
        'input_size': 384,
        'batch_size': 10,
        'learning_rate': 1e-5,
        'description': '🏥 Optimized for medical imaging tasks'
    },
    22: {
        'name': 'swin_large_patch4_window12_384',
        'display_name': 'Swin Large (384px) - Window Attention',
        'input_size': 384,
        'batch_size': 8,
        'learning_rate': 5e-6,
        'description': '🪟 Hierarchical window attention'
    },
    23: {
        'name': 'densenet121.tv_in1k',
        'display_name': 'DenseNet121 (ImageNet-21k Pre-trained)',
        'input_size': 224,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'description': '🌿 Dense connectivity, memory efficient, good for medical imaging'
    },
    24: {
        'name': 'mobilenetv1_100.ra4_e3600_r224_in1k',
        'display_name': 'MobileNet-V1 (224px) - Lightweight & Fast',
        'input_size': 224,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'description': '📱 Ultra-lightweight, perfect for edge deployment and fast training'
    }
}

def get_gpu_memory_gb() -> float:
    """Estimate available GPU memory in GB"""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0

def adjust_batch_size_for_gpu(config: Dict[str, Any], use_fp16: bool = True) -> Dict[str, Any]:
    """Adjust batch size based on available GPU memory"""
    gpu_memory = get_gpu_memory_gb()
    
    if gpu_memory == 0:
        # CPU only - reduce batch size significantly
        config = config.copy()
        config['batch_size'] = max(1, config['batch_size'] // 4)
        return config
    
    # Estimate memory usage
    input_size = config['input_size']
    base_batch_size = config['batch_size']
    
    # Memory estimation (rough)
    memory_per_sample = (input_size ** 2 * 3 * 4) / (1024**3)  # RGB image in GB
    if use_fp16:
        memory_per_sample *= 0.6  # FP16 uses less memory
    
    # Model memory (rough estimates)
    model_memory = {
        'small': 1,    # < 50M params
        'base': 2,     # 50-100M params  
        'large': 4,    # 100-300M params
        'huge': 8,     # > 300M params
        'xlarge': 12   # Very large models
    }
    
    # Estimate model size category
    if 'small' in config['name'] or 'eva02_base' in config['name']:
        model_mem = model_memory['small']
    elif 'base' in config['name']:
        model_mem = model_memory['base']
    elif 'large' in config['name']:
        model_mem = model_memory['large']
    elif 'xlarge' in config['name'] or 'huge' in config['name']:
        model_mem = model_memory['huge']
    else:
        model_mem = model_memory['base']
    
    # Calculate max batch size
    available_memory = gpu_memory * 0.8  # Leave 20% buffer
    max_batch_size = int((available_memory - model_mem) / memory_per_sample)
    
    # Adjust batch size
    adjusted_config = config.copy()
    adjusted_config['batch_size'] = min(base_batch_size, max(1, max_batch_size))
    
    return adjusted_config

def display_model_menu() -> int:
    """Display interactive model selection menu"""
    print("\n" + "="*80)
    print("🤖 MODEL SELECTION MENU - TimM Top ImageNet Models")
    print("="*80)
    
    # Group models by category
    categories = {
        "Current Defaults (Recommended for beginners)": [1, 2, 3, 23],
        "State-of-the-Art (Highest Accuracy)": [4, 5, 6, 7],
        "Modern CNNs (ConvNeXt Family)": [8, 11, 14, 17, 21],
        "Classic CNNs (Proven Architectures)": [23],
        "Lightweight Models (Fast Training & Deployment)": [24],
        "Vision Transformers (Advanced)": [9, 10, 15, 16],
        "EfficientNet Family (Balanced)": [1, 12, 18],
        "Hybrid Architectures (CNN + Transformer)": [13, 20],
        "Swin Transformers (Window Attention)": [3, 19, 22]
    }
    
    gpu_memory = get_gpu_memory_gb()
    print(f"🖥️  GPU Memory Available: {gpu_memory:.1f} GB")
    print()
    
    for category, model_ids in categories.items():
        print(f"📁 {category}:")
        for model_id in model_ids:
            if model_id in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_id]
                memory_req = "High VRAM" if config['batch_size'] <= 4 else "Medium VRAM" if config['batch_size'] <= 8 else "Low VRAM"
                
                # Add GPU compatibility indicator
                if gpu_memory >= 16 or config['batch_size'] >= 8:
                    compat = "✅"
                elif gpu_memory >= 8 or config['batch_size'] >= 4:
                    compat = "⚠️"
                else:
                    compat = "❌"
                
                print(f"  {compat} [{model_id:2d}] {config['display_name']}")
                print(f"      📊 Input: {config['input_size']}px | Batch: {config['batch_size']} | {memory_req}")
                print(f"      💡 {config['description']}")
        print()
    
    print("💡 Legend:")
    print("   ✅ = Recommended for your GPU")
    print("   ⚠️  = May need batch size adjustment")
    print("   ❌ = Requires high-end GPU")
    print()
    
    while True:
        try:
            choice = int(input("Select model number (1-24): "))
            if choice in MODEL_CONFIGS:
                return choice
            else:
                print(f"❌ Invalid choice. Please select between 1 and {max(MODEL_CONFIGS.keys())}")
        except ValueError:
            print("❌ Please enter a valid number")

def get_model_config(model_id: int, use_fp16: bool = True) -> Dict[str, Any]:
    """Get optimized configuration for selected model"""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Model ID {model_id} not found")
    
    config = MODEL_CONFIGS[model_id].copy()
    
    # Adjust for GPU memory
    config = adjust_batch_size_for_gpu(config, use_fp16)
    
    # Additional FP16 optimizations
    if use_fp16:
        # Slightly increase batch size for FP16 efficiency
        config['batch_size'] = min(config['batch_size'] + 2, config['batch_size'] * 1.3)
        config['batch_size'] = int(config['batch_size'])
    
    return config

def print_model_recommendations(config: Dict[str, Any], use_fp16: bool = True):
    """Print optimized settings for selected model"""
    print("\n" + "="*60)
    print("🎯 OPTIMIZED TRAINING SETTINGS")
    print("="*60)
    print(f"Model: {config['display_name']}")
    print(f"Architecture: {config['name']}")
    print(f"Description: {config['description']}")
    print()
    print("📊 Training Parameters:")
    print(f"  • Input Size: {config['input_size']}px")
    print(f"  • Batch Size: {config['batch_size']}")
    print(f"  • Learning Rate: {config['learning_rate']}")
    print(f"  • Precision: {'FP16' if use_fp16 else 'FP32'}")
    print()
    
    # Memory and performance estimates
    gpu_memory = get_gpu_memory_gb()
    if gpu_memory > 0:
        estimated_memory = (config['input_size'] ** 2 * 3 * config['batch_size'] * 4) / (1024**3)
        if use_fp16:
            estimated_memory *= 0.6
        
        print(f"💾 Estimated VRAM Usage: {estimated_memory:.1f} GB / {gpu_memory:.1f} GB available")
        
        if estimated_memory > gpu_memory * 0.9:
            print("⚠️  WARNING: May exceed GPU memory. Consider reducing batch size.")
        elif estimated_memory > gpu_memory * 0.7:
            print("⚠️  High memory usage. Monitor for OOM errors.")
        else:
            print("✅ Memory usage looks good!")
    
    print("="*60)
