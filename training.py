import os
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from roboflow import Roboflow
import matplotlib.pyplot as plt
import numpy as np
import logging
import warnings

# Suppress verbose warnings
warnings.filterwarnings("ignore", ".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", ".*copying from a non-meta parameter.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# --- Constants ---
CHECKPOINT = 'facebook/detr-resnet-50'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- COCO Dataset Class ---
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, annotation_file, image_processor):
        super(CocoDetection, self).__init__(img_folder, annotation_file)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        orig_size = torch.tensor([img.size[1], img.size[0]])
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        
        encoding = self.image_processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        target["orig_size"] = orig_size
        return pixel_values, target

# --- Collate Function ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return {'pixel_values': torch.empty(0), 'pixel_mask': torch.empty(0), 'labels': []}

    pixel_values = [item[0] for item in batch]
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# --- PyTorch Lightning Module for DETR ---
class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, id2label, batch_size, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        if freeze_backbone:
            for n, p in self.model.named_parameters():
                if "backbone" in n:
                    p.requires_grad = False

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.hparams.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}

# --- Loss Plotting Callback ---
class LossPlotCallback(Callback):
    def __init__(self, save_dir='training_plots'):
        super().__init__()
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch_idx = trainer.current_epoch + 1
        self.epochs.append(epoch_idx)
        tr = metrics.get("train/loss")
        self.train_losses.append(float(tr.cpu()) if tr is not None else np.nan)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        va = metrics.get("val/loss")
        if len(self.val_losses) < len(self.epochs):
             self.val_losses.append(float(va.cpu()) if va is not None else np.nan)


    def on_train_end(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)
        n = min(len(self.epochs), len(self.train_losses), len(self.val_losses) if self.val_losses else len(self.epochs))
        ep = self.epochs[:n]
        tr = self.train_losses[:n]
        va = self.val_losses[:n] if self.val_losses else [np.nan]*n

        plt.figure(figsize=(10,6))
        plt.plot(ep, tr, marker='.', label='train/loss')
        plt.plot(ep, va, marker='.', label='val/loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss per Epoch')
        plt.grid(True); plt.legend()
        plt.tight_layout()
        out = os.path.join(self.save_dir, 'loss_curves.png')
        plt.savefig(out)
        print(f"[LossPlotCallback] Saved: {out}")
        plt.close()
        
# --- Main Training Function ---
def train_model(api_key, workspace, project_name, version, epochs=10, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, freeze_backbone=True, batch_size=16, num_workers=4):
    """
    Orchestrates the DETR model training process.
    """
    # TODO: Handle API key securely
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset_version = project.version(version)
    dataset_location = dataset_version.download("coco").location

    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

    ANNOTATION_FILE_NAME = "_annotations.coco.json"
    train_folder = os.path.join(dataset_location, "train")
    val_folder = os.path.join(dataset_location, "valid")
    train_annotation_file = os.path.join(train_folder, ANNOTATION_FILE_NAME)
    val_annotation_file = os.path.join(val_folder, ANNOTATION_FILE_NAME)

    train_dataset = CocoDetection(img_folder=train_folder, annotation_file=train_annotation_file, image_processor=image_processor)
    val_dataset = CocoDetection(img_folder=val_folder, annotation_file=val_annotation_file, image_processor=image_processor)

    categories = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True if num_workers > 0 else False)

    model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay, id2label=id2label, batch_size=batch_size, freeze_backbone=freeze_backbone)

    loss_plot_callback = LossPlotCallback(save_dir="training_plots")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    csv_logger = CSVLogger(save_dir="lightning_logs", name="detr_run")
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="detr_tb")

    trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=epochs,
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_model_summary=True,
        callbacks=[loss_plot_callback, lr_monitor],
        logger=[csv_logger, tb_logger],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Save the final model weights and config
    final_model_path = "detr_isolated_trained_weights.pth"
    model_data = {
        'state_dict': model.state_dict(),
        'num_classes': len(id2label),
        'id2label': id2label
    }
    torch.save(model_data, final_model_path)
    
    return final_model_path, "training_plots/loss_curves.png"

if __name__ == '__main__':
    # Example of how to run the training
    # Replace with your actual Roboflow details
    API_KEY = "8LphsYHJxlPrbZc2rNfn"
    WORKSPACE = "runxy"
    PROJECT = "isolated-6chwu"
    VERSION = 2
    
    trained_model_path, loss_plot_path = train_model(
        api_key=API_KEY,
        workspace=WORKSPACE,
        project_name=PROJECT,
        version=VERSION,
        epochs=1 # Set to a small number for testing
    )
    print(f"Training complete. Model saved to: {trained_model_path}")
    print(f"Loss plot saved to: {loss_plot_path}")