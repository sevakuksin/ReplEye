import os
import cv2
import torch
import pandas as pd
from pathlib import Path

# Path to the trained YOLOv5 weights
weights = '../../vial_detection/runs/train/exp_augmented_glass2/weights/best.pt'
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)  # Replace 'best.pt' with your model's path

# Directory setup
input_dir = '../../saved_images/'  # Directory containing input images
output_dir = '../annotations/'  # Directory to save isolated vial images
os.makedirs(output_dir, exist_ok=True)

# CSV setup
csv_file = os.path.join(output_dir, 'volume_labels.csv')
annotations = []

# Process each image in the input directory
for image_file in os.listdir(input_dir):
    if image_file.endswith(('jpg', 'jpeg', 'png')):
        image_path = os.path.join(input_dir, image_file)

        # Perform inference
        results = model(image_path)
        detections = results.xyxy[0]  # Bounding box results (xmin, ymin, xmax, ymax, confidence, class)

        # Load image with OpenCV
        img = cv2.imread(image_path)

        # Process each detection
        for idx, det in enumerate(detections):
            xmin, ymin, xmax, ymax, conf, cls = det.tolist()
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            # Skip detections with low confidence
            if conf < 0.6:
                continue

            # Crop and save detected vial
            vial_img = img[ymin:ymax, xmin:xmax]
            vial_name = f"{Path(image_file).stem}_vial{idx}.png"
            vial_path = os.path.join(output_dir, vial_name)
            cv2.imwrite(vial_path, vial_img)

            # The volume of the liquid in the vial is in the name of the image: {volume}_{some number}.{extension}
            volume = Path(image_file).stem.split('_')[0]

            # Add annotation to list
            annotations.append({
                'filename': vial_name,
                'original_image': image_file,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'confidence': conf,
                'class': cls,
                'volume': volume
            })

# Save annotations to CSV
df = pd.DataFrame(annotations)
df.to_csv(csv_file, index=False)

print(f"Processing complete. Vials saved in '{output_dir}' and annotations in '{csv_file}'.")
