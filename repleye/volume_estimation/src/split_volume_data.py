import os
import random
import shutil


# Split data into train and val sets, there are no xml files, so just copy the csv file from annotations_dir to base_dir
# Do not take xml files, just copy 1 csv file from annotations_dir to base_dir
def create_directories(base_dir):
    """Creates train and validation directories if they don't exist, and clears them if they do."""
    for subset in ["train", "val"]:
            dir_path = os.path.join(base_dir, subset)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)  # Delete existing files
            os.makedirs(dir_path, exist_ok=True)


def split_data(annotations_dir, base_dir, volume_labels='volume_labels.csv', train_ratio=0.8):
    """
    Splits the images and annotations into train and val sets
    """
    create_directories(base_dir)
    # Copy train_ratio of the png files to train and the rest to val
    png_files = [f for f in os.listdir(annotations_dir) if f.endswith(".png")]
    random.shuffle(png_files)
    split_index = int(len(png_files) * train_ratio)
    train_files = png_files[:split_index]
    val_files = png_files[split_index:]
    for subset, files in zip(["train", "val"], [train_files, val_files]):
        for png_file in files:
            src_png_path = os.path.join(annotations_dir, png_file)
            dest_png_path = os.path.join(base_dir, subset, png_file)
            shutil.copy(src_png_path, dest_png_path)

    # Copy the csv file to base_dir
    shutil.copy(os.path.join(annotations_dir, volume_labels), os.path.join(base_dir, volume_labels))


if __name__ == "__main__":
    annotations_dir = os.path.abspath("../annotations")
    base_dir = os.path.abspath("../data")
    split_data(annotations_dir, base_dir, volume_labels='volume_labels.csv', train_ratio=0.8)