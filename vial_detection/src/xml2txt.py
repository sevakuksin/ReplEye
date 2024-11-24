import os
import xml.etree.ElementTree as Etree


def convert_voc_to_yolo(xml_file, classes, output_folder):
    tree = Etree.parse(xml_file)
    root = tree.getroot()
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(xml_file))[0] + ".txt")

    with open(output_file, "w") as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Write in YOLO format
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# Usage
# classes = ["vial"]  # Update this list with your class names
# xml_folder = "data/annotations"
# output_folder = "annotations"
# os.makedirs(output_folder, exist_ok=True)
#
# for xml_file in os.listdir(xml_folder):
#     if xml_file.endswith(".xml"):
#         convert_voc_to_yolo(os.path.join(xml_folder, xml_file), classes, output_folder)
