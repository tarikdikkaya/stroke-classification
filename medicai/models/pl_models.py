import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pytorch_lightning as pl
import numpy as np
import os
from typing import Optional, List, Union

class TimmLightningClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, learning_rate: float = 1e-3, 
                 input_size: int = 224, class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, mixup_alpha: float = 0.0):
        """
        Initialize the classifier with a timm model.
        
        Parameters:
            model_name (str): One of 'tf_efficientnet_b3_ns', 'tf_efficientnetv2_s', 
                              'vit_small_patch16_224', or 'resnest50d'.
            num_classes (int): Number of output classes.
            learning_rate (float): Learning rate for the optimizer.
            input_size (int): Input image size (square dimensions). Default: 224.
            class_weights (torch.Tensor): Class weights for imbalanced datasets.
            label_smoothing (float): Label smoothing factor.
            mixup_alpha (float): Mixup alpha parameter.
        """
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Loss function with optional label smoothing and class weights
        if label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.mixup_alpha = mixup_alpha
        
        # Store class weights for potential use
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation to input data."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Calculate mixup loss."""
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply mixup if enabled
        if self.mixup_alpha > 0:
            x, y_a, y_b, lam = self.mixup_data(x, y, self.mixup_alpha)
            logits = self(x)
            loss = self.mixup_criterion(logits, y_a, y_b, lam)
        else:
            logits = self(x)
            loss = self.loss_fn(logits, y)
            
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use different optimizers based on model type
        if 'vit' in self.model.__class__.__name__.lower() or hasattr(self.model, 'patch_embed'):
            # Vision Transformers work better with AdamW
            optimizer = optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate, 
                weight_decay=0.05,
                betas=(0.9, 0.999)
            )
        else:
            # CNNs can use SGD with momentum
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.learning_rate, 
                momentum=0.9, 
                weight_decay=1e-4,
                nesterov=True
            )
        
        # Use Cosine annealing with warm restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart period each time
            eta_min=1e-7  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def train_model(self, train_dataloader, val_dataloader, max_epochs: int = 10,
                   accumulate_grad_batches: int = 1, callbacks: Optional[List] = None,
                   precision: Union[str, int] = "32", logger_dir: Optional[str] = None):
        """
        Train the model using PyTorch Lightning Trainer with configurable precision.

        Parameters:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            max_epochs (int): Maximum number of training epochs.
            accumulate_grad_batches (int): Number of batches to accumulate gradients.
            callbacks (List, optional): Additional callbacks for training.
            precision (Union[str, int]): Training precision. Options: "32", "16-mixed", "bf16-mixed", 16, 32.
            logger_dir (str, optional): Directory for logging. If None, uses default logger.
        """
        # Default callbacks
        if callbacks is None:
            callbacks = [
                pl.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    mode='min',
                    save_top_k=2,
                    filename='{epoch}-{val_loss:.4f}'
                ),
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    mode='min'
                )
            ]
        
        # Configure model for the specified precision
        self.configure_model_for_precision(precision)
        
        # Configure gradient clipping for FP16 stability
        gradient_clip_val = 1.0 if precision in ["16-mixed", 16] else 0.0
        
        # Configure logger
        logger = None
        if logger_dir:
            from pytorch_lightning.loggers import TensorBoardLogger
            logger = TensorBoardLogger(save_dir=logger_dir, name="training_logs")
        else:
            # Disable default logger to avoid path issues
            logger = False
            
        trainer = pl.Trainer(
            precision=precision,
            max_epochs=max_epochs,
            callbacks=callbacks,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        trainer.fit(self, train_dataloader, val_dataloader)
        return trainer

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted class indices.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
        return preds

    def load(self, checkpoint_path: str) -> 'TimmLightningClassifier':
        """
        Load model weights from a checkpoint.

        Parameters:
            checkpoint_path (str): Path to the checkpoint file to load.
            
        Returns:
            self: The model instance with loaded weights.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            # Check if the file is a TorchScript model
            if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
                try:
                    # Try to load as TorchScript model
                    self.model = torch.jit.load(checkpoint_path, map_location=self.device)
                    print(f"Model successfully loaded from TorchScript file: {checkpoint_path}")
                    return self
                except Exception as e:
                    print(f"Failed to load as TorchScript model, trying regular checkpoint: {e}")
                    
            # If not a TorchScript model or loading as TorchScript failed, try regular PyTorch checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                self.load_state_dict(checkpoint)
                
            print(f"Model successfully loaded from checkpoint: {checkpoint_path}")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {str(e)}")

    def export(self, filepath: str, example_input: torch.Tensor = None):
        """
        Export the model to a TorchScript file.

        Parameters:
            filepath (str): Path to save the exported model.
            example_input (torch.Tensor): Example input for tracing. Defaults to a tensor with the shape [1, 3, input_size, input_size].
        """
        
        if example_input is None:
            example_input = torch.randn(1, 3, self.input_size, self.input_size)
        
        # Trace only the core model for inference efficiency
        scripted_model = torch.jit.trace(self.model, example_input)
        scripted_model.save(filepath)
        print(f"Model exported to {filepath}")

    def configure_model_for_precision(self, precision: Union[str, int]):
        """
        Configure the model for specific precision training.
        
        Parameters:
            precision: Training precision ("32", "16-mixed", "bf16-mixed", 16, 32)
        """
        if precision in ["16-mixed", 16]:
            # Enable automatic mixed precision optimizations
            # Convert batch norm to float32 for stability
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.float()
                # Ensure layer norm stays in float32 for stability
                elif isinstance(module, nn.LayerNorm):
                    module.float()
        
        return self
