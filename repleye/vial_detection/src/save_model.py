import sys
from pathlib import Path

# import cv2
import torch

yolov5_path = Path("../../yolov5")  # Adjust the path as needed
sys.path.append(str(yolov5_path))
#
# import torch
# from yolov5.models.common import DetectMultiBackend
#
# # Load the YOLOv5 model
weights = '../runs/train/exp_augmented_glass2/weights/best.pt'  # Path to your custom weights
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = DetectMultiBackend(weights, device=device)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
model = model.fuse()
# Save the model locally
torch.save(model, '../models/fused_model.pt')

print("Model saved as 'fused_model.pt'")

#
# # Load the pre-fused model
# # model = torch.load('../models/fused_model.pt')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
#
# # Ensure the model is in evaluation mode
# model.eval()
#
# # Run inference
# img = cv2.imread('../annotations/vials/5_0.jpeg')
# # to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # make it torch tensor
# img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
# # img = img  # Convert [B, H, W, C] -> [B, C, H, W]
#
# results = model(img)  # Provide the path to an image
# results.print()  # Print detected objects
# results.save()  # Save output images with detections
