import os
import random
import shutil

from xml2txt import convert_voc_to_yolo


def create_directories(base_dir):
    """Creates train and validation directories if they don't exist, and clears them if they do."""
    for subset in ["train", "val"]:
        for folder in ["images", "labels"]:
            dir_path = os.path.join(base_dir, subset, folder)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)  # Delete existing files
            os.makedirs(dir_path, exist_ok=True)


def split_data(annotations_dir, base_dir, classes, train_ratio=0.8):
    """
    Splits the images and annotations into train and val sets and
    converts XML annotations to YOLO format.

    Parameters:
    annotations_dir (str): Path to the annotations and images.
    base_dir (str): Path where train and val directories will be created.
    train_ratio (float): Ratio of training data, e.g., 0.8 for 80% train, 20% val.
    """
    create_directories(base_dir)

    # Collect all XML files
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    random.shuffle(xml_files)

    # Split data into training and validation
    split_index = int(len(xml_files) * train_ratio)
    train_files = xml_files[:split_index]
    val_files = xml_files[split_index:]

    for subset, files in zip(["train", "val"], [train_files, val_files]):
        for xml_file in files:
            # Define paths for images and annotations
            image_file = xml_file.replace(".xml", ".jpg")
            src_image_path = os.path.join(annotations_dir, image_file)
            src_xml_path = os.path.join(annotations_dir, xml_file)
            dest_image_path = os.path.join(base_dir, subset, "images", image_file)
            dest_txt_dir = os.path.join(base_dir, subset, "labels")
            # Copy image file to destination
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dest_image_path)

            # Convert XML to YOLO format and save as .txt
            convert_voc_to_yolo(src_xml_path, classes=classes, output_folder=dest_txt_dir)

    print(f"Data split complete! Train set: {len(train_files)} files, Val set: {len(val_files)} files.")


def background(background_dir, base_dir, num_files=2):
    """Randomly selects background files and copies them to train and val directories."""
    background_files = [f for f in os.listdir(background_dir) if f.endswith(".jpg")]
    random.shuffle(background_files)
    train_background = background_files[:num_files]
    val_background = background_files[num_files:]

    for subset, files in zip(["train", "val"], [train_background, val_background]):
        for bg_file in files:
            src_bg_path = os.path.join(background_dir, bg_file)
            dest_bg_path = os.path.join(base_dir, subset, "images", bg_file)
            shutil.copy(src_bg_path, dest_bg_path)

    print(f"Background images copied to train and val directories.")


# Parameters
if __name__ == "__main__":
    annotations_dir = "../annotations/vials"  # Path to your original images and XML files
    base_dir = "../../data"  # Base directory where train/val folders will be created

    # Run the script
    split_data(annotations_dir, base_dir, classes=["vial"], train_ratio=0.8)

    # Randomly choose 2 background files for each training and validation data
    background_dir = "../annotations/background"
    background(background_dir, base_dir, num_files=2)
