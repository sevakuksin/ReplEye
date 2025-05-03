# Author-
# Description-

import adsk.cam
import adsk.core
import adsk.fusion
import csv
import os
import traceback
import random


def capture_screenshot(filename, eye_position, target_position, up_vector):
    app = adsk.core.Application.get()
    viewport = app.activeViewport
    camera = viewport.camera

    # Set up the camera properties
    camera.isPerspective = True  # Perspective view for a more realistic angle
    camera.eye = eye_position
    camera.target = target_position
    camera.upVector = up_vector

    # Apply the new camera settings
    viewport.camera = camera
    viewport.refresh()

    # Save the screenshot
    viewport.saveAsImageFile(filename, 1920, 1080)


# Function to generate random distortions
def randomize_camera(base_camera, max_offset=4, up_offset=0.1):
    # Create copies of base camera's eye and target positions with random offsets
    eye_x = base_camera.eye.x + random.uniform(-max_offset, max_offset)
    eye_y = base_camera.eye.y + random.uniform(-max_offset, max_offset)
    eye_z = base_camera.eye.z + random.uniform(-max_offset, max_offset)
    eye_position = adsk.core.Point3D.create(eye_x, eye_y, eye_z)

    target_x = base_camera.target.x + random.uniform(-max_offset, max_offset)
    target_y = base_camera.target.y + random.uniform(-max_offset, max_offset)
    target_z = base_camera.target.z + random.uniform(-max_offset, max_offset)
    target_position = adsk.core.Point3D.create(target_x, target_y, target_z)

    up_x = base_camera.upVector.x + random.uniform(-up_offset, up_offset)
    up_y = base_camera.upVector.y + random.uniform(-up_offset, up_offset)
    up_z = base_camera.upVector.z + random.uniform(-up_offset, up_offset)
    up_vector = adsk.core.Vector3D.create(up_x, up_y, up_z)

    return eye_position, target_position, up_vector


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    viewport = app.activeViewport
    base_camera = viewport.camera

    try:
        # Set up parameters
        design = app.activeProduct
        parameters = design.userParameters
        extrusion_param = parameters.itemByName('water_height')
        body = design.rootComponent.bRepBodies.item(0)  # Assuming single body
        output_folder = os.path.expanduser("~/FusionScreenshots")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        n = 4  # Number of samples
        d = 3  # Number of distortions
        height_increment = 40 / n  # Change in extrusion height per step
        csv_path = os.path.join(output_folder, "volume_labels.csv")

        # Open CSV file for writing labels
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "volume"])

            for _ in range(d):
                # Iterate through each extrusion height
                eye_position, target_position, up_vector = randomize_camera(base_camera)

                # Set up the distorted camera
                camera = viewport.camera
                camera.isPerspective = True
                camera.eye = eye_position
                camera.target = target_position
                camera.upVector = up_vector  # Keep the same upVector as the base

                # Apply the camera settings
                viewport.camera = camera
                viewport.refresh()

                for i in range(1, n + 1):
                    new_height = i * height_increment
                    extrusion_param.expression = f'{new_height} mm'
                    volume = body.volume  # Get current volume

                    # Take screenshot
                    filename = f'ExtrusionHeight_{new_height}mm_{d}_offset.png'
                    file_path = os.path.join(output_folder, filename)
                    capture_screenshot(file_path, eye_position, target_position, up_vector)

                    # Write volume label to CSV
                    writer.writerow([filename, volume])

        ui.messageBox("Data generation completed successfully.")

    except Exception as e:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
