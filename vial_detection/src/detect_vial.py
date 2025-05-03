import cv2
import torch
import numpy as np
# from pathlib import Path

def detect(image_path_or_image, weights, confidence_threshold=0.4, model=None):
    """
    Detects objects in an image using a YOLOv5 model.

    Args:
        image_path_or_image: str, numpy.ndarray, or torch.Tensor
            Path to the input image, the image as a numpy array, or a PyTorch tensor.
        weights: str
            Path to the YOLOv5 weights file.
        confidence_threshold: float
            Minimum confidence score to consider a detection valid.
        model: torch.nn.Module

    Returns:
        List[Dict]:
            A list of annotations with details about detected objects.
            Each annotation includes xmin, ymin, xmax, ymax, confidence, and class.
    """
    # Load YOLOv5 model
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # Handle different input types
    if isinstance(image_path_or_image, str):  # Image path
        img = cv2.imread(image_path_or_image)
        if img is None:
            raise ValueError(f"Could not read image from path: {image_path_or_image}")
    elif isinstance(image_path_or_image, np.ndarray):  # NumPy array
        img = image_path_or_image
    elif isinstance(image_path_or_image, torch.Tensor):  # PyTorch tensor
        img = image_path_or_image
    else:
        raise TypeError("Input must be a file path (str), numpy array, or torch.Tensor")

    # Perform inference
    results = model(img)

    # Retrieve bounding box results
    detections = results.xyxy[0]  # Format: [xmin, ymin, xmax, ymax, confidence, class]
    annotations = []

    # Process each detection
    for det in detections:
        xmin, ymin, xmax, ymax, conf, cls = det.tolist()
        if conf >= confidence_threshold:
            annotations.append({
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'confidence': conf,
                'class': cls
            })

    return annotations
