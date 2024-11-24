import subprocess
import os
import sys

weights = "yolov5s.pt"

# Define arguments
args = [
    sys.executable, "yolov5/train.py",
    "--img", "640",
    "--batch", "4",
    "--epochs", "220",
    "--data", "vial_detection/config/dataset.yaml",
    "--weights", weights,
    "--project", "vial_detection/runs/train",
    "--name", "exp_augmented_glass",
    "--cache", "disk",
    "--hyp", "vial_detection/config/hyperparameters.yaml"
]

# Change directory to vial_detection root
os.chdir('../../')
# Run YOLOv5 training with subprocess
subprocess.run(args)