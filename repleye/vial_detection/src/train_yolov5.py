import subprocess
import os
import sys

# weights = "yolov5s.pt"
weights = "yolov5n.pt"

# Define arguments
args = [
    sys.executable, "yolov5/train.py",
    "--img", "640",
    "--batch", "16",
    "--epochs", "100",
    "--data", "vial_detection/config/dataset.yaml",
    "--weights", weights,
    "--cfg", "vial_detection/config/yolov5n.yaml",
    "--project", "vial_detection/runs/train",
    "--name", "new_data",
    "--cache", "disk",
    "--hyp", "vial_detection/config/hyperparameters.yaml"
]

# Change directory to vial_detection root
os.chdir('../../')
# Run YOLOv5 training with subprocess
subprocess.run(args)