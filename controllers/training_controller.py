"""
Controller untuk training operations
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from roboflow import Roboflow

from models.detr_model import Detr, CocoDetection, LossPlotCallback
from models.training_config import TrainingConfig


def collate_fn(batch):
    """
    Collate function untuk DataLoader
    """
    from transformers import DetrImageProcessor
    
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return {'pixel_values': torch.empty(0), 'pixel_mask': torch.empty(0), 'labels': []}

    pixel_values = [item[0] for item in batch]
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


class TrainingController:
    """
    Controller untuk mengelola training process
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize TrainingController
        
        Args:
            config: TrainingConfig object
        """
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.trainer = None
    
    def prepare_dataset(self):
        """
        Download dan prepare dataset dari Roboflow
        
        Returns:
            tuple: (train_loader, val_loader, id2label)
        """
        CHECKPOINT = 'facebook/detr-resnet-50'
        
        # Download dataset dari Roboflow
        rf = Roboflow(api_key=self.config.api_key)
        project = rf.workspace(self.config.workspace).project(self.config.project_name)
        dataset_version = project.version(self.config.version)
        dataset_location = dataset_version.download("coco").location

        # Prepare image processor
        image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

        # Prepare dataset paths
        ANNOTATION_FILE_NAME = "_annotations.coco.json"
        train_folder = os.path.join(dataset_location, "train")
        val_folder = os.path.join(dataset_location, "valid")
        train_annotation_file = os.path.join(train_folder, ANNOTATION_FILE_NAME)
        val_annotation_file = os.path.join(val_folder, ANNOTATION_FILE_NAME)

        # Create datasets
        train_dataset = CocoDetection(
            img_folder=train_folder, 
            annotation_file=train_annotation_file, 
            image_processor=image_processor
        )
        val_dataset = CocoDetection(
            img_folder=val_folder, 
            annotation_file=val_annotation_file, 
            image_processor=image_processor
        )

        # Extract categories
        categories = train_dataset.coco.cats
        id2label = {k: v['name'] for k, v in categories.items()}
        
        # Create data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=self.config.num_workers, 
            pin_memory=True, 
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        self.val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=self.config.num_workers, 
            pin_memory=True, 
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        return self.train_loader, self.val_loader, id2label
    
    def initialize_model(self, id2label: dict):
        """
        Initialize DETR model
        
        Args:
            id2label: Mapping dari id ke label
        """
        self.model = Detr(
            lr=self.config.lr,
            lr_backbone=self.config.lr_backbone,
            weight_decay=self.config.weight_decay,
            id2label=id2label,
            batch_size=self.config.batch_size,
            freeze_backbone=self.config.freeze_backbone
        )
        return self.model
    
    def initialize_trainer(self):
        """
        Initialize PyTorch Lightning Trainer
        
        Returns:
            pl.Trainer instance
        """
        loss_plot_callback = LossPlotCallback(save_dir="training_plots")
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        csv_logger = CSVLogger(save_dir="lightning_logs", name="detr_run")
        tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="detr_tb")

        self.trainer = pl.Trainer(
            devices=1, 
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=self.config.epochs,
            gradient_clip_val=0.1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            enable_model_summary=True,
            callbacks=[loss_plot_callback, lr_monitor],
            logger=[csv_logger, tb_logger],
        )
        return self.trainer
    
    def train(self) -> tuple:
        """
        Execute training process
        
        Returns:
            tuple: (final_model_path, loss_plot_path)
        """
        # Prepare dataset
        train_loader, val_loader, id2label = self.prepare_dataset()
        
        # Initialize model
        model = self.initialize_model(id2label)
        
        # Initialize trainer
        trainer = self.initialize_trainer()
        
        # Start training
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Save model
        final_model_path = "detr_isolated_trained_weights.pth"
        model_data = {
            'state_dict': model.state_dict(),
            'num_classes': len(id2label),
            'id2label': id2label
        }
        torch.save(model_data, final_model_path)
        
        loss_plot_path = "training_plots/loss_curves.png"
        
        return final_model_path, loss_plot_path
