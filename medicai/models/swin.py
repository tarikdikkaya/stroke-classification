"""Swin Transformer model implementation for medical image classification using PyTorch Lightning."""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import timm
from torch.cuda.amp import autocast
import pytorch_lightning as pl

from .base_lightning_module import BaseMedicalModel

logger = logging.getLogger(__name__)

class SwinLarge(BaseMedicalModel):
    """
    Swin Transformer Large model for medical image classification with PyTorch Lightning.
    
    This model inherits from BaseMedicalModel for Lightning integration.
    """
    
    def __init__(
        self, 
        num_classes: int = 2,
        pretrained: bool = True,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        input_size: int = 384,
        max_epochs: int = 100,
    ):
        """
        Initialize Swin Transformer Large model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            class_weights: Optional tensor of class weights for loss function
            input_size: Input image size
            max_epochs: Maximum training epochs
        """
        # Initialize the base Lightning module
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=class_weights,
            max_epochs=max_epochs
        )
        
        # Create model backbone
        self.backbone = timm.create_model(
            "swin_large_patch4_window12_384",
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.input_size = input_size
    
    @torch.jit.export
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        return self.backbone.forward_features(x)
    
    def predict_with_probs(
        self, 
        image_tensor: torch.Tensor,
        return_probs: bool = True
    ) -> Union[int, Tuple[int, float]]:
        """
        Perform inference on a preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor [1, C, H, W]
            return_probs: Whether to return class probabilities
            
        Returns:
            Predicted class index or tuple of (class index, probability)
        """
        # Make sure model is in eval mode
        self.eval()
        
        with torch.no_grad():
            with autocast():
                outputs = self(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
        _, predicted = torch.max(probs, 1)
        predicted_idx = predicted.item()
        
        if return_probs:
            return predicted_idx, probs[0][predicted_idx].item()
        return predicted_idx