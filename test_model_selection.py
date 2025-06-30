#!/usr/bin/env python3
"""
Test script for the interactive model selection system
"""

from model_configs import display_model_menu, get_model_config, print_model_recommendations

def test_model_selection():
    """Test the interactive model selection"""
    print("🧪 Testing Model Selection System")
    print("="*50)
    
    # Test automatic model selection
    print("\n1. Testing automatic model selection (model ID 1):")
    config = get_model_config(1, use_fp16=True)
    print_model_recommendations(config, use_fp16=True)
    
    print("\n2. Testing high-performance model (model ID 4):")
    config = get_model_config(4, use_fp16=True)
    print_model_recommendations(config, use_fp16=True)
    
    print("\n3. Testing interactive menu:")
    print("(This will show the full interactive menu)")
    # Uncomment the next line to test interactive menu
    # model_id = display_model_menu()
    # config = get_model_config(model_id, use_fp16=True)
    # print_model_recommendations(config, use_fp16=True)
    
    print("\n✅ Model selection system test completed!")

if __name__ == "__main__":
    test_model_selection()
