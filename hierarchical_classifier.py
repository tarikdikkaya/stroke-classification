"""
Hierarchical Stroke Classification System
This module implements a two-stage classification system for stroke detection.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    primary_class: str
    confidence: float
    secondary_classes: List[Tuple[str, float]]
    final_prediction: List[str]
    ct_uncertainty: Optional[float] = None
    mri_uncertainty: Optional[float] = None
    requires_review: bool = False

class HierarchicalStrokeClassifier:
    """
    Two-stage hierarchical classifier for stroke detection:
    Stage 1: CT Binary Classification (NormalKronik vs Other)
    Stage 2: MRI Binary Classification (Hiperakut vs Subakut)
    """
    
    def __init__(self, 
                 ct_model_path: str, 
                 mri_model_path: str,
                 ct_threshold: float = 0.7,
                 mri_threshold: float = 0.6,
                 confidence_threshold: float = 0.5,
                 uncertainty_threshold: float = 0.3):
        
        self.ct_model = self.load_model(ct_model_path)
        self.mri_model = self.load_model(mri_model_path)
        
        # Thresholds for different stages
        self.ct_threshold = ct_threshold
        self.mri_threshold = mri_threshold
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        # Class mappings
        self.ct_classes = ['NormalKronik', 'Other']
        self.mri_classes = ['Hiperakut', 'Subakut']
        
        logger.info(f"Initialized HierarchicalStrokeClassifier with thresholds:")
        logger.info(f"CT: {ct_threshold}, MRI: {mri_threshold}, Confidence: {confidence_threshold}")
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load a trained model from checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    
    def calculate_uncertainty(self, probs: torch.Tensor) -> float:
        """Calculate prediction uncertainty using entropy."""
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item()
    
    def classify(self, ct_image: torch.Tensor, mri_image: torch.Tensor) -> ClassificationResult:
        """
        Enhanced hierarchical classification with confidence-based decision making
        """
        with torch.no_grad():
            # Stage 1: CT Classification
            ct_logits = self.ct_model(ct_image.unsqueeze(0))
            ct_probs = F.softmax(ct_logits, dim=1)
            ct_confidence, ct_pred = torch.max(ct_probs, dim=1)
            
            ct_class = self.ct_classes[ct_pred.item()]
            ct_conf = ct_confidence.item()
            ct_uncertainty = self.calculate_uncertainty(ct_probs[0])
            
            # Stage 2: MRI Classification
            mri_logits = self.mri_model(mri_image.unsqueeze(0))
            mri_probs = F.softmax(mri_logits, dim=1)
            mri_uncertainty = self.calculate_uncertainty(mri_probs[0])
            
            # Enhanced decision logic
            final_classes = []
            secondary_classes = []
            requires_review = False
            
            if ct_class == 'NormalKronik' and ct_conf >= self.ct_threshold:
                # High confidence normal - but still check MRI for subtle findings
                final_classes.append('NormalKronik')
                
                # Check if MRI shows any acute findings above threshold
                for i, mri_class in enumerate(self.mri_classes):
                    mri_class_conf = mri_probs[0, i].item()
                    if mri_class_conf >= self.mri_threshold:
                        final_classes.append(f"Possible_{mri_class}")
                        secondary_classes.append((mri_class, mri_class_conf))
                        requires_review = True
            
            elif ct_class == 'Other' or ct_conf < self.ct_threshold:
                # Abnormal CT or low confidence - rely more on MRI
                mri_confidence, mri_pred = torch.max(mri_probs, dim=1)
                primary_mri_class = self.mri_classes[mri_pred.item()]
                primary_mri_conf = mri_confidence.item()
                
                # Add primary MRI prediction
                final_classes.append(primary_mri_class)
                
                # Check secondary MRI class
                other_mri_idx = 1 - mri_pred.item()
                other_mri_conf = mri_probs[0, other_mri_idx].item()
                
                if other_mri_conf >= self.confidence_threshold:
                    secondary_class = self.mri_classes[other_mri_idx]
                    final_classes.append(f"Also_Consider_{secondary_class}")
                    secondary_classes.append((secondary_class, other_mri_conf))
            
            # Check for high uncertainty cases
            if (ct_uncertainty > self.uncertainty_threshold or 
                mri_uncertainty > self.uncertainty_threshold):
                requires_review = True
                final_classes.append("High_Uncertainty")
            
            return ClassificationResult(
                primary_class=ct_class,
                confidence=ct_conf,
                secondary_classes=secondary_classes,
                final_prediction=final_classes,
                ct_uncertainty=ct_uncertainty,
                mri_uncertainty=mri_uncertainty,
                requires_review=requires_review
            )
    
    def batch_classify(self, ct_images: torch.Tensor, mri_images: torch.Tensor) -> List[ClassificationResult]:
        """Classify a batch of images."""
        results = []
        for i in range(ct_images.size(0)):
            result = self.classify(ct_images[i], mri_images[i])
            results.append(result)
        return results
    
    def ensemble_predict(self, ct_images: List[torch.Tensor], mri_images: List[torch.Tensor]) -> ClassificationResult:
        """
        Ensemble prediction using multiple image slices/views
        """
        results = []
        for ct_img, mri_img in zip(ct_images, mri_images):
            result = self.classify(ct_img, mri_img)
            results.append(result)
        
        # Aggregate predictions using weighted voting
        return self._aggregate_predictions(results)
    
    def _aggregate_predictions(self, results: List[ClassificationResult]) -> ClassificationResult:
        """Aggregate multiple predictions using confidence-weighted voting."""
        # Implementation of ensemble aggregation
        # This is a simplified version - can be enhanced based on specific needs
        
        if not results:
            raise ValueError("No results to aggregate")
        
        # Find the result with highest confidence
        best_result = max(results, key=lambda x: x.confidence)
        
        # Count votes for each class
        class_votes = {}
        total_confidence = 0
        
        for result in results:
            for pred_class in result.final_prediction:
                if pred_class not in class_votes:
                    class_votes[pred_class] = 0
                class_votes[pred_class] += result.confidence
                total_confidence += result.confidence
        
        # Normalize votes
        for class_name in class_votes:
            class_votes[class_name] /= total_confidence
        
        # Create aggregated result
        aggregated_classes = [class_name for class_name, vote in class_votes.items() 
                            if vote >= 0.3]  # Threshold for inclusion
        
        return ClassificationResult(
            primary_class=best_result.primary_class,
            confidence=best_result.confidence,
            secondary_classes=best_result.secondary_classes,
            final_prediction=aggregated_classes,
            ct_uncertainty=np.mean([r.ct_uncertainty for r in results]),
            mri_uncertainty=np.mean([r.mri_uncertainty for r in results]),
            requires_review=any(r.requires_review for r in results)
        )

def create_hierarchical_datasets(base_path: str) -> Tuple[Path, Path]:
    """
    Prepare datasets for hierarchical training
    """
    base_path = Path(base_path)
    
    # Dataset 1: CT Binary Classification (NormalKronik vs Other)
    ct_dataset_path = base_path / "ct_binary_dataset"
    ct_dataset_path.mkdir(exist_ok=True)
    
    # Create CT training structure
    (ct_dataset_path / "train" / "NormalKronik").mkdir(parents=True, exist_ok=True)
    (ct_dataset_path / "train" / "Other").mkdir(parents=True, exist_ok=True)
    (ct_dataset_path / "val" / "NormalKronik").mkdir(parents=True, exist_ok=True)
    (ct_dataset_path / "val" / "Other").mkdir(parents=True, exist_ok=True)
    
    # Dataset 2: MRI Binary Classification (Hiperakut vs Subakut)  
    mri_dataset_path = base_path / "mri_binary_dataset"
    mri_dataset_path.mkdir(exist_ok=True)
    
    # Create MRI training structure
    (mri_dataset_path / "train" / "Hiperakut").mkdir(parents=True, exist_ok=True)
    (mri_dataset_path / "train" / "Subakut").mkdir(parents=True, exist_ok=True)
    (mri_dataset_path / "val" / "Hiperakut").mkdir(parents=True, exist_ok=True)
    (mri_dataset_path / "val" / "Subakut").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created dataset structures at:")
    logger.info(f"CT Dataset: {ct_dataset_path}")
    logger.info(f"MRI Dataset: {mri_dataset_path}")
    
    return ct_dataset_path, mri_dataset_path

if __name__ == "__main__":
    # Example usage
    base_path = "a:/Teknofest SYZ/ASAMA_2_DATASETS/Yarışma 2.aşama veri seti kümesi"
    ct_path, mri_path = create_hierarchical_datasets(base_path)
    
    print(f"Datasets prepared at:")
    print(f"CT: {ct_path}")
    print(f"MRI: {mri_path}")
