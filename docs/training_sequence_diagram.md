sequenceDiagram
    participant User
    participant TrainingView
    participant TrainingController
    participant Roboflow as Roboflow API
    participant Detr as Detr Model
    participant Trainer as PL Trainer

    User->>TrainingView: Submit Form (API Key, Workspace, Project, Version, Epochs, Batch Size)
    TrainingView->>TrainingView: Validate Inputs (api_key, epochs, batch_size)

    Note over TrainingView: Initialize Config
    TrainingView->>TrainingController: __init__(config: TrainingConfig)
    TrainingController-->>TrainingView: controller instance

    TrainingView->>TrainingController: train()
    
    TrainingController->>Roboflow: Roboflow(api_key)
    Roboflow-->>TrainingController: dataset_location
    
    TrainingController->>TrainingController: prepare_dataset()
    
    Note over TrainingController: Create datasets and dataloaders internaly
    TrainingController-->>TrainingController: returns (train_loader, val_loader, id2label)

    Note over TrainingController: Phase 2 - Initialize Model

    TrainingController->>Detr: __init__(lr, lr_backbone, weight_decay, id2label, batch_size, freeze_backbone)

    
    Detr-->>TrainingController: model instance


    Note over TrainingController: Phase 3 - Initialize Trainer
    TrainingController->>TrainingController: initialize_trainer()
    
    Note over TrainingController: Initialize Callbacks & Loggers
    TrainingController->>Trainer: __init__(max_epochs, accelerator, devices, callbacks, logger)
    Trainer-->>TrainingController: trainer instance


    Note over TrainingController: Phase 4 - Training Loop
    TrainingController->>Trainer: fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    loop Every Epoch
        loop Training/Validation Batch
            Trainer->>Detr: training_step(batch, batch_idx)
            Detr->>Detr: forward(pixel_values, pixel_mask)
            Detr-->>Trainer: loss
        end
    end
    
    Trainer-->>TrainingController: Training Completed

    Note over TrainingController: Save Final Model
    TrainingController->>TrainingController: torch.save(model_data, final_model_path)

    TrainingController-->>TrainingView: returns (final_model_path, loss_plot_path)

    TrainingView->>User: Display Success Message
    TrainingView->>User: Show Download Button & Loss Plot