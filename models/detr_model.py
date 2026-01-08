"""
Model classes untuk DETR
"""
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import numpy as np
import os
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


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Custom COCO Detection Dataset class
    """
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


class Detr(pl.LightningModule):
    """
    PyTorch Lightning Module untuk DETR model
    """
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
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=self.hparams.batch_size)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False, 
                    batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=self.hparams.batch_size)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, 
                    batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() 
                       if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() 
                       if "backbone" in n and p.requires_grad], 
             "lr": self.hparams.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.hparams.lr, 
                                     weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}


class LossPlotCallback(Callback):
    """
    Callback untuk plotting loss selama training
    """
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
        n = min(len(self.epochs), len(self.train_losses), 
                len(self.val_losses) if self.val_losses else len(self.epochs))
        ep = self.epochs[:n]
        tr = self.train_losses[:n]
        va = self.val_losses[:n] if self.val_losses else [np.nan]*n

        plt.figure(figsize=(10,6))
        plt.plot(ep, tr, marker='.', label='train/loss')
        plt.plot(ep, va, marker='.', label='val/loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(self.save_dir, 'loss_curves.png')
        plt.savefig(out)
        print(f"[LossPlotCallback] Saved: {out}")
        plt.close()
