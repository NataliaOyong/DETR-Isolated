import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import warnings

# Import Hugging Face transformers DETR
try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Please install: pip install transformers")

def load_model(weights_path, num_classes=2, device='cpu', checkpoint='facebook/detr-resnet-50'):
    """
    Load DETR model with pretrained weights for isolated mathematical expression detection.
    This function loads a Hugging Face transformers DETR model that was trained with PyTorch Lightning.
    
    Args:
        weights_path: Path to the model weights file
        num_classes: Number of classes (1 for background + 1 for mathematical expression)
        device: Device to load model on ('cpu' or 'cuda')
        checkpoint: Hugging Face checkpoint name (default: 'facebook/detr-resnet-50')
    
    Returns:
        Tuple of (model, image_processor) - both loaded and ready for inference
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    device = torch.device(device)
    
    # Load image processor
    try:
        image_processor = DetrImageProcessor.from_pretrained(checkpoint)
    except Exception as e:
        raise Exception(f"Error loading image processor: {e}")
    
    # Initialize model
    try:
        print(f"Initializing DETR model from checkpoint: {checkpoint}...")
        model = DetrForObjectDetection.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        raise Exception(f"Error initializing model: {e}")
    
    # Load checkpoint weights
    try:
        checkpoint_data = torch.load(weights_path, map_location=device)
    except Exception as e:
        raise Exception(f"Error loading checkpoint file: {e}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint_data, dict):
        # Check if it's a PyTorch Lightning checkpoint (has 'state_dict' key)
        if 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        # Check if model is saved directly
        elif 'model' in checkpoint_data and isinstance(checkpoint_data['model'], torch.nn.Module):
            print("Loading complete model from checkpoint...")
            model = checkpoint_data['model']
            model.to(device)
            model.eval()
            return model, image_processor
        else:
            # Assume it's a state_dict itself
            state_dict = checkpoint_data
    else:
        # Assume it's a state_dict (OrderedDict)
        state_dict = checkpoint_data
    
    # Process state_dict keys
    # PyTorch Lightning saves with 'model.' prefix because model is wrapped in LightningModule
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'model.' prefix if present (from PyTorch Lightning wrapper)
        if k.startswith('model.'):
            name = k[6:]  # Remove 'model.' prefix
        # Remove 'module.' prefix if present (from DataParallel)
        elif k.startswith('module.'):
            name = k[7:]  # Remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    
    # Load state dict into model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys not found in model (first 5: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys (first 5: {unexpected_keys[:5]})")
        print("Model weights loaded successfully!")
    except Exception as e:
        raise Exception(f"Error loading state dict: {e}")
    
    model.to(device)
    model.eval()
    
    return model, image_processor


def predict_isolated_expressions(model, image_processor, image, confidence_threshold=0.5, device='cpu'):
    """
    Predict isolated mathematical expressions in an image using Hugging Face DETR.
    
    Args:
        model: Loaded DetrForObjectDetection model
        image_processor: DetrImageProcessor instance
        image: PIL Image
        confidence_threshold: Minimum confidence score for detections
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        List of dictionaries containing:
        - 'bbox': bounding box coordinates [x1, y1, x2, y2]
        - 'confidence': confidence score
        - 'label': label id
        - 'cropped_image': PIL Image of the cropped detection
    """
    device = torch.device(device)
    
    # Preprocess image using DetrImageProcessor
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_mask = inputs["pixel_mask"].to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    # Post-process results
    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # [height, width]
    
    results = image_processor.post_process_object_detection(
        outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes
    )[0]
    
    # Extract predictions
    detections = []
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Process each detection
    for box, score, label in zip(boxes, scores, labels):
        # Box format from DETR is [x1, y1, x2, y2] (normalized or absolute)
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image bounds and valid
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_width, int(x2))
        y2 = min(img_height, int(y2))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop the detected region
        cropped = image.crop((x1, y1, x2, y2))
        
        # Skip if cropped image is too small
        if cropped.size[0] < 5 or cropped.size[1] < 5:
            continue
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(score),
            'label': int(label),
            'cropped_image': cropped
        })
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detections
