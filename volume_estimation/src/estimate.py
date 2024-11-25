# Use the model to predict the volume of a vial from an image:

import torch
# from torchvision import transforms
from PIL import Image
from model import VolumeEstimator
from transformations import val_transforms
# Load the model

def estimate_volume(image_path, model_path):
    model = VolumeEstimator()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image = Image.open(image_path)
    transform = val_transforms
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        volume = model(image)
    return volume.item()

if __name__ == "__main__":
    image_path = '../data/val/17_0_vial6.png'
    model_path = '../models/model_2024_11_24.pth'
    volume = estimate_volume(image_path, model_path)
    print(f"Estimated volume: {volume} ml")





