import subprocess
import os
import sys
from matplotlib import pyplot as plt
from PIL import Image

image = "annotations/vials/img (14).jpg"
weights = "runs/train/exp_augmented_1/weights/best.pt"

# Define detection arguments
args = [
    sys.executable, "yolov5/detect.py",  # Path to detect.py in YOLOv5
    "--source", image,  # Path to the image you want to detect objects in
    "--weights", weights,            # Path to YOLOv5 weights
    "--img", "640",                       # Image size
    "--conf", "0.5",                      # Confidence threshold
    "--project", "runs/detect",           # Directory to save results
    "--name", "exp_detection",            # Experiment name for output folder
    "--save-txt",                         # Save results in text files
    "--save-conf",                        # Save confidence scores
]

# Change directory to Repleye root
os.chdir('../')  # Adjust if needed

# Run YOLOv5 detection with subprocess
subprocess.run(args)

# # Path to the output image
output_image_path = "runs/detect/exp_detection/" + os.path.basename(image)

# Display the output image with detections
if os.path.exists(output_image_path):
    image = Image.open(output_image_path)
    plt.imshow(image)
    plt.axis("off")  # Turn off axis
    plt.show()
else:
    print("Detection output image not found.")

# python yolov5/detect.py --weights runs/train/exp_augmented_1/weights/best.pt --source videos/video_vials.mp4 --conf-thres 0.6 --output videos/ --save-txt --save-conf --project runs/detect --name exp_detection_video