import cv2
import torch

from repleye.vial_detection import detect_vial
from repleye.volume_estimation.src import estimate
from repleye.volume_estimation.src.model import VolumeEstimator


# from PIL import Image


def analyze_image(image, yolo_weights, volume_weights, yolo_model=None, volume_model=None):
    """
    Analyze an image to detect vials and estimate their volumes.

    Args:
        image: str or ndarray or tensor
            Path to the input image or the image itself.
        yolo_weights: str
            Path to the YOLO weights for vial detection.
        volume_weights: str
            Path to the model weights for volume estimation.
        yolo_model: torch.nn.Module
            A YOLOv5 model for vial detection.
        volume_model: torch.nn.Module
            A volume estimation model.

    Returns:
        List[Dict]:
            A list of dictionaries, each containing vial information
            (bounding box, volume estimate, and position).
    """
    # Detect vials in the image
    vials = detect_vial.detect(image, yolo_weights, model=yolo_model)  # Returns list of annotations

    # Sort vials from left to right by the horizontal center of the bounding box
    sorted_vials = sorted(vials, key=lambda vial: (vial['xmin'] + vial['xmax']) / 2)

    # Estimate the volume of each vial
    results = []
    for vial in sorted_vials:
        # Crop the vial image using bbox coordinates (if needed for volume estimation)
        vial_image = extract_vial_image(image, vial)

        # Estimate volume
        volume = estimate.estimate_volume(vial_image, volume_weights, volume_model)

        # Add information to results
        results.append({
            'bbox': vial,
            'volume': volume,
            'center_x': (vial['xmin'] + vial['xmax']) / 2
        })

    return results


def extract_vial_image(image, bbox):
    """
    Extract a cropped image of the vial based on the bounding box.

    Args:
        image: str or ndarray or tensor
            Path to the input image or the image itself.
        bbox: Dict
            A dictionary containing bbox coordinates: xmin, ymin, xmax, ymax.

    Returns:
        numpy.ndarray:
            The cropped image of the vial.
    """

    # Load the image
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

    # cv2.imshow('image', img[ymin:ymax, xmin:xmax])
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.destroyAllWindows()

    # Crop the image
    return img[ymin:ymax, xmin:xmax]


if __name__ == "__main__":
    image_path = 'vial_detection/annotations/vials/5_1.jpeg'
    yolo_weights = 'vial_detection/runs/train/exp_augmented_glass2/weights/best.pt'
    volume_weights = 'volume_estimation/models/model_2024_11_24.pth'

    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)
    volume_model = VolumeEstimator()
    volume_model.load_state_dict(torch.load(volume_weights))
    volume_model.eval()

    results = analyze_image(image_path, yolo_weights, volume_weights, yolo_model, volume_model)

    for idx, result in enumerate(results):
        print(f"Vial {idx + 1}: Volume = {result['volume']:.2f} ml, Center X = {result['center_x']}")
