# import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class VolumeEstimator(nn.Module):
    def __init__(self, weights=MobileNet_V2_Weights.IMAGENET1K_V1):
        super(VolumeEstimator, self).__init__()
        # Load MobileNetV2 without the classifier layer
        self.base_model = models.mobilenet_v2(weights=weights)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier[1].in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for regression
        )

    def forward(self, x):
        return self.base_model(x)
