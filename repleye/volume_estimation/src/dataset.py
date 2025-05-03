import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformations import resize_transform


# from torchvision import transforms

class VolumeDataset(Dataset):
    """Loader for volume estimation dataset, returns paired images and volumes lists."""

    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all images (e.g., 'data/train').
            csv_file (string): Path to the CSV file with filenames and volumes.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load the CSV file
        csv_path = csv_file
        self.labels_df = pd.read_csv(csv_path)
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        # print(self.image_files[idx])
        # print(self.labels_df.loc[self.labels_df['filename'] == self.image_files[idx], 'volume'])
        volume = self.labels_df.loc[self.labels_df['filename'] == self.image_files[idx], 'volume'].values[0]
        volume = torch.tensor(volume, dtype=torch.float32)

        # Apply the transformation
        if self.transform:
            image = self.transform(image)
        else:
            image = resize_transform(image)

        return image, volume

    def get_data(self):
        """Returns all the image files and volumes 2 lists"""
        images = []
        volumes = []
        for idx in range(len(self.image_files)):
            image, volume = self.__getitem__(idx)
            images.append(image)
            volumes.append(volume)

# Usage example
# dataset = VolumeDataset(root_dir='data', csv_file='volume_labels.csv')
# image, volume = dataset[0]
# print(image, volume)
