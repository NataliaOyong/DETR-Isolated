"""
Data classes untuk training configuration
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration class untuk training parameters
    """
    api_key: str
    workspace: str
    project_name: str
    version: int
    epochs: int = 10
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    freeze_backbone: bool = True
    batch_size: int = 16
    num_workers: int = 4
    
    def __post_init__(self):
        """Validasi config setelah initialization"""
        if not self.api_key:
            raise ValueError("API key tidak boleh kosong")
        if self.epochs < 1:
            raise ValueError("Epochs harus >= 1")
        if self.batch_size < 1:
            raise ValueError("Batch size harus >= 1")
