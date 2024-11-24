from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop to 224x224 and resize
    transforms.RandomHorizontalFlip(),  # Horizontal flip with 50% probability
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomRotation(10),  # Rotate by up to 15 degrees
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a validation/test transform without augmentations
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a transform for just resizing and converting to a tensor
resize_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])