import torch
import torch.optim as optim
import torch.nn as nn
import val
from torch.utils.data import DataLoader
# from torchvision import transforms
from model import VolumeEstimator  # Importing from model.py
from dataset import VolumeDataset  # Assuming you have a custom dataset class in dataset.py
from transformations import train_transforms, val_transforms  # Assuming you have defined transforms in transformations.py

def train_model(model, train_loader, val_loader, epochs, learning_rate, save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = 0
        model.train()
        for images, volumes in train_loader:
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions.squeeze(), volumes)
            loss.backward()
            optimizer.step()

        # Optionally validate and save the model at checkpoints
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        val.validate_model(model, val_loader)
        # Save model checkpoint here if needed
        torch.save(model.state_dict(), save_path)
        print(f"Model checkpoint saved at {save_path}")


if __name__ == "__main__":
    # Initialize model, data loaders, and hyperparameters
    train_dir = '../data/train'
    val_dir = '../data/val'
    labels = '../data/volume_labels.csv'
    save_path = '../models/model_checkpoint.pth'

    model = VolumeEstimator()
    train_loader = DataLoader(VolumeDataset(train_dir, labels, train_transforms), batch_size=8, shuffle=True)
    val_loader = DataLoader(VolumeDataset(val_dir, labels, val_transforms), batch_size=8)
    train_model(model, train_loader, val_loader, epochs=150, learning_rate=5e-4, save_path=save_path)
