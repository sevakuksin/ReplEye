import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

import val
from dataset import VolumeDataset
from model import VolumeEstimator
from transformations import train_transforms, val_transforms


def train_model(model, train_loader, val_loader, epochs, learning_rate, run):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best = np.inf
    train_losses = []
    val_losses = []
    save_path = run + 'last.pth'
    save_best = run + 'best.pth'


    for epoch in range(epochs):
        epoch_loss = 0  # Initialize total loss for the epoch
        model.train()
        for images, volumes in train_loader:
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions.squeeze(), volumes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate the batch loss

        # Calculate the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        # Validate and save the model at checkpoints
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")
        val_loss = val.validate_model(model, val_loader)[0]

        train_losses.append(avg_epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best checkpoint saved at {save_best}")
    # Save model checkpoint here if needed
    torch.save(model.state_dict(), save_path)
    print(f"Last model checkpoint saved at {save_path}")

    return train_losses, val_losses


if __name__ == "__main__":
    # Initialize model, data loaders, and hyperparameters
    train_dir = '../data/train'
    val_dir = '../data/val'
    labels = '../data/volume_labels.csv'
    run = 'run1'
    save_run = f'../runs/train/{run}/'
    # Create a run directory
    os.makedirs(save_run, exist_ok=True)

    model = VolumeEstimator()
    train_loader = DataLoader(VolumeDataset(train_dir, labels, train_transforms), batch_size=8, shuffle=True)
    val_loader = DataLoader(VolumeDataset(val_dir, labels, val_transforms), batch_size=8)
    losses = train_model(model, train_loader, val_loader, epochs=70, learning_rate=1e-4, run=save_run)

    # Save losses in a text file
    with open(save_run + 'losses.txt', 'w') as f:
        f.write('Train Losses\n')
        for loss in losses[0]:
            f.write(f'{loss}\n')
        f.write('Validation Losses\n')
        for loss in losses[1]:
            f.write(f'{loss}\n')

    # Save graph of losses
    plt.figure()
    plt.title('Loss during training')
    plt.plot(losses[0], '.-', label='Train Loss', color='r')
    plt.plot(losses[1], '.-', label='Validation Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_run + 'loss.png')
