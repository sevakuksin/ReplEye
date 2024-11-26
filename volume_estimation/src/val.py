import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformations import val_transforms
from dataset import VolumeDataset  # Import the custom dataset class
from model import VolumeEstimator  # Import the model from model.py


def validate_model(model, val_loader):
    # Set the model to evaluation mode
    model.eval()

    criterion = nn.MSELoss()
    val_loss = 0
    num_batches = 0
    losses = []
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, volumes in val_loader:
            # Run the model on the validation images
            predictions = model(images)

            # Compute the loss
            loss = criterion(predictions.squeeze(), volumes)
            val_loss += loss.item()
            losses.append(loss.item())
            num_batches += 1

    # Calculate average validation loss
    avg_val_loss = val_loss / num_batches
    print(f"Validation Loss: {avg_val_loss}")

    return avg_val_loss, losses


if __name__ == "__main__":
    # Path to the validation data and labels
    val_dir = '../data/val'
    labels = '../data/volume_labels.csv'
    run = 'run1'
    save_run = f'../runs/val/{run}/'
    model_path = '../models/model_2024_11_24.pth'

    # Load the trained model
    model = VolumeEstimator()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"No model checkpoint found at {model_path}. Please train the model first.")

    # Set up the validation data loader
    val_loader = DataLoader(VolumeDataset(val_dir, labels, val_transforms), batch_size=4, shuffle=False)

    # Run validation
    avg, losses = validate_model(model, val_loader)

    # Save the validation results
    os.makedirs(save_run, exist_ok=True)
    with open(save_run + 'val_losses.txt', 'w') as f:
        f.write('\n'.join(map(str, losses)))
    print(f"Validation losses saved at {save_run}val_losses.txt")
