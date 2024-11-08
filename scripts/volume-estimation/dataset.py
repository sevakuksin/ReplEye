import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# from torchvision import transforms

class VolumeDataset(Dataset):
    """Loader for volume estimation dataset, returns paired images and volumes lists."""

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all images (e.g., 'data/train').
            csv_file (string): Path to the CSV file with filenames and volumes.
        """
        self.root_dir = root_dir

        # Load the CSV file
        csv_path = csv_file
        self.labels_df = pd.read_csv(csv_path)
        self.image_files = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a fixed size
            transforms.ToTensor()]) # Convert to a tensor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)
        volume = self.labels_df.loc[self.labels_df['filename'] == self.image_files[idx], 'volume'].values[0]
        volume = torch.tensor(volume, dtype=torch.float32)
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
