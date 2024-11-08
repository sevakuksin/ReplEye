import subprocess
import os
import sys

weights = "yolov5s.pt"

# Define arguments
args = [
    sys.executable, "yolov5/train.py",
    "--img", "640",
    "--batch", "2",
    "--epochs", "200",
    "--data", "config/dataset.yaml",
    "--weights", weights,
    "--project", "runs/train",
    "--name", "exp_augmented_1",
    "--cache", "disk",
    "--hyp", "config/hyperparameters.yaml"
]

# Change directory to Repleye root
os.chdir('../')
# Run YOLOv5 training with subprocess
subprocess.run(args)