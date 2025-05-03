import cv2
import torch
import time
import sys
import os

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from repleye.analyze import analyze_image # adjust import as needed
from repleye.volume_estimation.src.model import VolumeEstimator

if Picamera2 is None:
    print("This script requires Picamera2 and should be run on a Raspberry Pi.")
    exit()

def stream():
    # Load models once
    yolo_weights = 'vial_detection/models/model_03_05_25.pt'
    volume_weights = 'volume_estimation/models/model_2024_11_24.pth'

    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)
    yolo_model.eval()

    volume_model = VolumeEstimator()
    volume_model.load_state_dict(torch.load(volume_weights, map_location='cpu'))
    volume_model.eval()

    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # warm-up

    print("Starting live preview. Press 'q' to quit.")

    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run analysis
        results = analyze_image(
            image=rgb_frame,
            yolo_weights=yolo_weights,
            volume_weights=volume_weights,
            yolo_model=yolo_model,
            volume_model=volume_model
        )

        # Annotate frame
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
            volume = result['volume']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{volume:.2f} ml"
            cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Vial Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    picam2.close()
    cv2.destroyAllWindows()
    print("Preview ended.")

# Example usage
if __name__ == "__main__":
    stream()
