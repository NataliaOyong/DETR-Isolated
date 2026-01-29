sequenceDiagram
    participant User
    participant DetectionView
    participant DetectionController
    participant ModelUtils
    participant ImageProcessor
    participant DetrModel

    Note over User, DetectionView: User visits Detection Page

    User->>DetectionView: Upload Image/PDF
    DetectionView->>DetectionView: process_uploaded_file(uploaded_file)
    DetectionView-->>DetectionView: return image
    
    User->>DetectionView: Click "Deteksi"
    DetectionView->>DetectionView: render_detection_section(image, detection_controller)
    
    DetectionView->>DetectionController: detect(image, confidence_threshold)
    
    DetectionController->>ModelUtils: predict_isolated_expressions(model, image_processor, image, threshold, device)
    
    Note over ModelUtils: 1. Preprocessing
    ModelUtils->>ImageProcessor: __call__(images=image, return_tensors="pt")
    ImageProcessor-->>ModelUtils: inputs (pixel_values, pixel_mask)
    ModelUtils->>ModelUtils: pixel_values.to(device)
    ModelUtils->>ModelUtils: pixel_mask.to(device)
    
    Note over ModelUtils: 2. Inference
    ModelUtils->>DetrModel: forward(pixel_values, pixel_mask)
    DetrModel-->>ModelUtils: outputs (logits, pred_boxes)
    
    Note over ModelUtils: 3. Postprocessing
    ModelUtils->>ModelUtils: target_sizes = tensor([image.size])
    ModelUtils->>ImageProcessor: post_process_object_detection(outputs, threshold, target_sizes)
    ImageProcessor-->>ModelUtils: results (boxes, scores, labels)
    
    Note over ModelUtils: 4. Cropping & Formatting
    loop For each detection
        ModelUtils->>ModelUtils: Calculate coordinates (x1, y1, x2, y2)
        ModelUtils->>ModelUtils: cropped = image.crop((x1, y1, x2, y2))
        
        alt size >= 5x5
            ModelUtils->>ModelUtils: Add to detections list
        end
    end
    ModelUtils->>ModelUtils: Sort detections by confidence desc
    
    ModelUtils-->>DetectionController: detections (list of dicts)
    
    DetectionController-->>DetectionView: detections
    
    Note over DetectionView: Display Results
    DetectionView->>DetectionView: draw_boxes_on_image(image, detections)
    DetectionView->>DetectionView: render_results_section(detections)
    DetectionView->>DetectionView: render_export_section(detections)
    
    DetectionView->>User: Show Result Image & Download Buttons