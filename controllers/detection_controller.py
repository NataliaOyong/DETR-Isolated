"""
Controller untuk detection/inference operations
"""
from PIL import Image
from typing import List, Dict
from utils.model_utils import predict_isolated_expressions


class DetectionController:
    """
    Controller untuk mengelola detection operations
    """
    
    def __init__(self, model, image_processor, device: str = 'cpu', confidence_threshold: float = 0.5):
        """
        Initialize DetectionController
        
        Args:
            model: Loaded DETR model
            image_processor: Image processor untuk preprocessing
            device: Device untuk inference
            confidence_threshold: Threshold untuk confidence score
        """
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.confidence_threshold = confidence_threshold
    
    def detect(self, image: Image.Image, confidence_threshold: float = None) -> List[Dict]:
        """
        Detect isolated expressions dalam gambar
        
        Args:
            image: PIL Image object
            confidence_threshold: Optional threshold override
            
        Returns:
            List of detection dictionaries dengan keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - label: int
            - cropped_image: PIL Image
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        detections = predict_isolated_expressions(
            self.model,
            self.image_processor,
            image,
            confidence_threshold=threshold,
            device=self.device
        )
        
        return detections
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold harus antara 0.0 dan 1.0")
        self.confidence_threshold = threshold
