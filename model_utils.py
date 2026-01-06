import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import logging
import warnings

# Suppress verbose warnings and logs
warnings.filterwarnings("ignore", ".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", ".*copying from a non-meta parameter.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import Hugging Face transformers DETR
try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Please install: pip install transformers")

def load_model(weights_path, device='cpu', checkpoint='facebook/detr-resnet-50'):
    """
    Load DETR model with pretrained weights.
    Handles new format (dict with config) and old format (state_dict only).
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    device = torch.device(device)
    
    # Load image processor
    image_processor = DetrImageProcessor.from_pretrained(checkpoint)
    
    # Load checkpoint data to CPU first, allowing for pickled objects (the config dict)
    checkpoint_data = torch.load(weights_path, map_location='cpu')
    
    num_classes = 2 # Default for old model format
    state_dict = None

    # Handle new and old formats
    if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
        print("Loading model from new format (with config)...")
        num_classes = checkpoint_data.get('num_classes', 2)
        state_dict = checkpoint_data['state_dict']
    else:
        print("Loading model from old format (state_dict only)...")
        state_dict = checkpoint_data
    
    # Initialize model with the determined number of classes
    print(f"Initializing DETR model from checkpoint: {checkpoint} with {num_classes} classes...")
    model = DetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Process state_dict keys for compatibility
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('model.', '', 1)
        new_state_dict[name] = v
    
    # Load state dict into model
    model.load_state_dict(new_state_dict, strict=False, assign=True)
    
    # Move the materialized model to the target device
    model.to(device)
    model.eval()
    print("Model weights loaded successfully!")
    
    return model, image_processor

def predict_isolated_expressions(model, image_processor, image, confidence_threshold=0.5, device='cpu'):
    """
    Predict isolated mathematical expressions in an image using Hugging Face DETR.
    """
    device = torch.device(device)
    
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_mask = inputs["pixel_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    
    results = image_processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes
    )[0]
    
    detections = []
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    img_width, img_height = image.size
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_width, int(x2))
        y2 = min(img_height, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        cropped = image.crop((x1, y1, x2, y2))
        
        if cropped.size[0] < 5 or cropped.size[1] < 5:
            continue
            
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(score),
            'label': int(label),
            'cropped_image': cropped
        })
    
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections
