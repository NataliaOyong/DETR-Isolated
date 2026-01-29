classDiagram
    %% Classes definitions

    class TrainingView {
        +render()
    }

    class TrainingController {
        +TrainingConfig config
        +DataLoader train_loader
        +DataLoader val_loader
        +Detr model
        +Trainer trainer
        +__init__(config: TrainingConfig)
        +prepare_dataset() tuple
        +initialize_model(id2label: dict) Detr
        +initialize_trainer() Trainer
        +train() tuple
    }

    class TrainingConfig {
        +str api_key
        +str workspace
        +str project_name
        +int version
        +int epochs
        +float lr
        +float lr_backbone
        +float weight_decay
        +bool freeze_backbone
        +int batch_size
        +int num_workers
        +__post_init__()
    }

    class Detr {
        +DetrForObjectDetection model
        +__init__(lr, lr_backbone, weight_decay, id2label, batch_size, freeze_backbone)
        +forward(pixel_values, pixel_mask)
        +common_step(batch)
        +training_step(batch, batch_idx)
        +validation_step(batch, batch_idx)
        +configure_optimizers() dict
    }

    class LossPlotCallback {
        +str save_dir
        +list train_losses
        +list val_losses
        +list epochs
        +__init__(save_dir: str)
        +on_train_epoch_end(trainer, pl_module)
        +on_validation_epoch_end(trainer, pl_module)
        +on_train_end(trainer, pl_module)
    }

    class Roboflow {
        +workspace(name)
        +project(name)
        +version(num)
        +download(format)
    }

    class Trainer {
        +fit(model, train_dataloaders, val_dataloaders)
    }

    %% Relationships
    TrainingView ..> "1" TrainingController : creates
    TrainingController "1" o-- "1" TrainingConfig : has
    TrainingController ..> "1" Roboflow : uses
    TrainingController "1" *-- "1" Detr : manages
    TrainingController ..> "1" LossPlotCallback : creates
    TrainingController "1" *-- "1" Trainer : manages
    
    %% Inheritance
    Detr --|> LightningModule
    LossPlotCallback --|> Callback

    %% External libraries (simplified representation)
    class LightningModule {
        <<interface>>
    }
    class Callback {
        <<interface>>
    }