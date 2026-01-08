"""
Controller untuk model operations (loading, saving)
"""
import torch
from utils.model_utils import load_model


class ModelController:
    """
    Controller untuk mengelola model DETR
    """
    
    def __init__(self, weights_path: str, device: str = 'cpu', checkpoint: str = 'facebook/detr-resnet-50'):
        """
        Initialize ModelController
        
        Args:
            weights_path: Path ke file weights model
            device: Device untuk inference ('cpu' atau 'cuda')
            checkpoint: Checkpoint name untuk HuggingFace
        """
        self.weights_path = weights_path
        self.device = device
        self.checkpoint = checkpoint
        self.model = None
        self.image_processor = None
        self._loaded = False
    
    def load(self):
        """
        Load model dan image processor
        
        Returns:
            tuple: (model, image_processor)
        """
        if not self._loaded:
            self.model, self.image_processor = load_model(
                self.weights_path, 
                device=self.device, 
                checkpoint=self.checkpoint
            )
            self._loaded = True
        return self.model, self.image_processor
    
    def is_loaded(self):
        """Check apakah model sudah dimuat"""
        return self._loaded
    
    def save_model(self, model, save_path: str, num_classes: int, id2label: dict):
        """
        Save model weights dan config
        
        Args:
            model: Model instance
            save_path: Path untuk menyimpan
            num_classes: Jumlah classes
            id2label: Mapping dari id ke label
        """
        model_data = {
            'state_dict': model.state_dict(),
            'num_classes': num_classes,
            'id2label': id2label
        }
        torch.save(model_data, save_path)
        return save_path
