# Use the model to predict the volume of a vial from an image:

import torch
# from torchvision import transforms
from PIL import Image
from model import VolumeEstimator
from transformations import val_transforms
# Load the model

model = VolumeEstimator()
model.load_state_dict(torch.load('../models/model_checkpoint.pth'))
model.eval()

# Load an image
image_path = '../annotations/ExtrusionHeight_74.0mm_0_offset.png'
image = Image.open(image_path)

transform = val_transforms

image = image.convert("RGB")
image = transform(image).unsqueeze(0)

# Predict the volume
with torch.no_grad():
    volume = model(image)

print(f"Predicted volume: {volume.item()}ml")



