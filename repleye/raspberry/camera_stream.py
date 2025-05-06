from flask import Flask, Response
from picamera2 import Picamera2
import time
import torch
import cv2
from repleye.analyze import analyze_image
from repleye.volume_estimation.src.model import VolumeEstimator
import os
import site

app = Flask(__name__)
picam2 = None
yolo_model = None
volume_model = None

# Define model paths - use site-packages location
site_packages = site.getsitepackages()[0]
YOLO_WEIGHTS = os.path.join(site_packages, 'repleye', 'vial_detection', 'models', 'model_03_05_25.pt')
VOLUME_WEIGHTS = os.path.join(site_packages, 'repleye', 'volume_estimation', 'models', 'model_2024_11_24.pth')

def init_models():
    global yolo_model, volume_model
    print("\n=== Initializing Models ===")
    print(f"Loading YOLO model from {YOLO_WEIGHTS}")
    try:
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_WEIGHTS)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        
    print(f"Loading volume model from {VOLUME_WEIGHTS}")
    try:
        volume_model = VolumeEstimator()
        volume_model.load_state_dict(torch.load(VOLUME_WEIGHTS))
        volume_model.eval()
        print("Volume model loaded successfully")
    except Exception as e:
        print(f"Error loading volume model: {e}")
    print("=== Models Initialized ===\n")

def init_camera():
    global picam2
    if picam2 is None:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration()
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Give camera time to start

def get_frame():
    init_camera()
    
    # Capture frame
    picam2.capture_file("frame.jpg")
    frame = cv2.imread("frame.jpg")
    print("\n=== Processing Frame ===")
    print(f"Frame captured, shape: {frame.shape}")
    
    # Analyze frame
    try:
        results = analyze_image(
            image=frame,
            yolo_weights=YOLO_WEIGHTS,
            volume_weights=VOLUME_WEIGHTS,
            yolo_model=yolo_model,
            volume_model=volume_model
        )
        print(f"Analysis complete: {len(results)} vials detected")
        
        # Annotate frame with results
        for i, result in enumerate(results):
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            volume = result['volume']
            print(f"Vial {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), volume={volume:.2f}ml")

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the volume label
            label = f"{volume:.2f} ml"
            label_position = (x1, y2 + 20)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    # Save annotated frame
    cv2.imwrite("frame.jpg", frame)
    print("Frame processed and saved\n")
    
    with open("frame.jpg", "rb") as f:
        return f.read()

@app.route('/')
def index():
    return '<html><body><img src="/stream" width="640" height="480"></body></html>'

@app.route('/stream')
def stream():
    def generate():
        while True:
            frame = get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)  # Limit to ~10 FPS
            
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Initialize models once at startup
    init_models()
    app.run(host='0.0.0.0', port=5001, debug=True) 