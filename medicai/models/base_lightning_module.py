"""Base Lightning module for all medical image classification models."""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, Precision, Recall, F1Score
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseMedicalModel(pl.LightningModule):
    """Base Lightning module that all medical image models will inherit from."""
    
    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        max_epochs: int = 100,
    ):
        """
        Initialize base model with common parameters.
        
        Args:
            num_classes: Number of output classes
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            class_weights: Optional weights for imbalanced classes
            max_epochs: Maximum number of training epochs
        """
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.max_epochs = max_epochs
        
        # Model backbone should be defined in child classes
        self.backbone = None
        
        # Save hyperparameters for easy checkpointing
        self.save_hyperparameters()
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Additional metrics for validation and test
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (to be implemented by child classes)."""
        if self.backbone is None:
            raise NotImplementedError("Model backbone not implemented")
        return self.backbone(x)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
    
    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        images, labels = batch
        outputs = self(images)
        
        # Loss calculation
        if self.class_weights is not None:
            # Make sure class_weights is on the same device
            if self.class_weights.device != outputs.device:
                self.class_weights = self.class_weights.to(outputs.device)
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(outputs, labels)
        
        # Calculate and log accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        images, labels = batch
        outputs = self(images)
        
        # Loss calculation
        if self.class_weights is not None:
            # Make sure class_weights is on the same device
            if self.class_weights.device != outputs.device:
                self.class_weights = self.class_weights.to(outputs.device)
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
            
        loss = criterion(outputs, labels)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        precision = self.val_precision(preds, labels)
        recall = self.val_recall(preds, labels)
        f1 = self.val_f1(preds, labels)
        
        # Log all metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", precision, on_epoch=True, logger=True)
        self.log("val_recall", recall, on_epoch=True, logger=True)
        self.log("val_f1", f1, on_epoch=True, logger=True)
        
        return {"val_loss": loss, "val_acc": acc}
    
    def test_step(self, batch, batch_idx):
        """Lightning test step."""
        images, labels = batch
        outputs = self(images)
        
        # Calculate metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)
        precision = self.test_precision(preds, labels)
        recall = self.test_recall(preds, labels)
        f1 = self.test_f1(preds, labels)
        
        # Loss calculation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        
        # Log metrics
        self.log("test_loss", loss, on_epoch=True, logger=True)
        self.log("test_acc", acc, on_epoch=True, logger=True)
        self.log("test_precision", precision, on_epoch=True, logger=True)
        self.log("test_recall", recall, on_epoch=True, logger=True)
        self.log("test_f1", f1, on_epoch=True, logger=True)
        
        return {"test_loss": loss, "test_acc": acc}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Lightning predict step."""
        if isinstance(batch, tuple) and len(batch) == 2:
            # If batch contains both images and labels (from a DataLoader)
            images, _ = batch
        else:
            # If batch is just the images
            images = batch
            
        outputs = self(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return {"predictions": preds, "probabilities": probs}
        
    @staticmethod
    def get_callbacks(
        monitor: str = "val_loss",
        mode: str = "min",
        early_stopping_patience: int = 10,
        checkpoint_filename: str = "best_model",
        save_top_k: int = 1,
    ) -> List[pl.Callback]:
        """
        Get standard Lightning callbacks for training.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            early_stopping_patience: Patience for early stopping
            checkpoint_filename: Filename pattern for checkpoints
            save_top_k: Number of best models to save
            
        Returns:
            List of callbacks
        """
        checkpoint_callback = ModelCheckpoint(
            filename=f"{checkpoint_filename}-{{epoch:02d}}-{{{monitor}:.4f}}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            verbose=True
        )
        
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            mode=mode,
            verbose=True
        )
        
        return [checkpoint_callback, early_stopping]

    def train_model(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        max_epochs=None,
        lr=None,
        weight_decay=None,
        class_weights=None,
        checkpoint_dir="checkpoints",
        early_stopping_patience=10,
        precision=16,  # Lightning's way to enable mixed precision
        gradient_clip_val=0,
        accumulate_grad_batches=1,
        resume_from_checkpoint=None,
        gpus=None,  # Auto-detect by default
        log_every_n_steps=50,
        *args, **kwargs
    ):
        """
        Train the model using PyTorch Lightning's Trainer.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: Optional DataLoader for test data
            max_epochs: Maximum number of training epochs
            lr: Learning rate (overrides the one set in __init__)
            weight_decay: Weight decay (overrides the one set in __init__)
            class_weights: Class weights for imbalanced datasets
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs to wait for improvement
            precision: Precision for training (16, 32, or 'bf16')
            gradient_clip_val: Gradient clipping value
            accumulate_grad_batches: Gradient accumulation steps
            resume_from_checkpoint: Path to checkpoint to resume from
            gpus: Number of GPUs to use (None for auto-detect)
            log_every_n_steps: How often to log metrics
            
        Returns:
            Dictionary with trainer and results
        """
        # Update model parameters if provided
        if lr is not None:
            self.learning_rate = lr
        if weight_decay is not None:
            self.weight_decay = weight_decay
        if class_weights is not None:
            self.class_weights = class_weights
        if max_epochs is not None:
            self.max_epochs = max_epochs
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Configure callbacks
        callbacks = self.get_callbacks(
            monitor="val_loss",
            mode="min",
            early_stopping_patience=early_stopping_patience,
            checkpoint_filename=os.path.join(checkpoint_dir, "best_model"),
            save_top_k=1
        )
        
        # Add a checkpoint callback that saves at the end of each epoch
        checkpoint_every_epoch = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="checkpoint-{epoch:02d}",
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=1,
            save_on_train_epoch_end=True
        )
        callbacks.append(checkpoint_every_epoch)
        
        # Configure Logger
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=os.path.join(checkpoint_dir, "logs"))
        
        # Configure Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=logger,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=log_every_n_steps,
        )
        
        # Train the model
        trainer.fit(self, train_loader, val_loader)
        
        # Test if test_loader provided
        test_results = None
        if test_loader is not None:
            test_results = trainer.test(self, test_loader)
        
        # Load best model before returning
        try:
            best_model_path = trainer.checkpoint_callback.best_model_path
            if best_model_path and os.path.exists(best_model_path):
                logger.info(f"Loading best model from {best_model_path}")
                best_model = self.load_from_checkpoint(best_model_path)
                self.load_state_dict(best_model.state_dict())
        except:
            logger.warning("Could not load best model. Using the last model state.")
        
        return {
            "trainer": trainer,
            "best_model_path": trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else None,
            "test_results": test_results
        }

    
    def evaluate(self, dataloader, criterion=None, device=None, use_amp=False):
        """
        Evaluate the model on a given dataloader.
        
        Args:
            dataloader: DataLoader with validation/test data
            criterion: Loss function (defaults to CrossEntropyLoss if None)
            device: Device to evaluate on (defaults to model's device if None)
            use_amp: Whether to use automatic mixed precision
            
        Returns:
            Tuple of (average_loss, accuracy_percentage)
        """
        if device is None:
            device = next(self.parameters()).device
            
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast() if use_amp else torch.no_grad():
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy