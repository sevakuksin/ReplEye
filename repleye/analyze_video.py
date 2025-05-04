import cv2
import torch
from repleye.analyze import analyze_image
from repleye.volume_estimation.src.model import VolumeEstimator


def process_video(input_video, output_video, yolo_weights, volume_weights):
    # Load YOLO and volume estimation models
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)
    volume_model = VolumeEstimator()
    volume_model.load_state_dict(torch.load(volume_weights))
    volume_model.eval()

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process video frame by frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}...")

        # Convert frame to RGB for YOLO model compatibility
        rgb_frame = frame #  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze frame
        results = analyze_image(
            image= rgb_frame,  # Provide the frame directly
            yolo_weights=yolo_weights,
            volume_weights=volume_weights,
            yolo_model=yolo_model,
            volume_model=volume_model
        )

        # Annotate frame with results
        for result in results:
            bbox = result['bbox']
            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            volume = result['volume']

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the volume label
            label = f"{volume:.2f} ml"
            label_position = (x1, y2 + 20)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write annotated frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print("Video processing complete. Output saved to:", output_video)


if __name__ == "__main__":
    input_video = 'vial_detection/videos/video_2.mp4'  # Path to input video
    output_video = 'output_video_2_1.avi'  # Path to save annotated video
    yolo_weights = 'vial_detection/runs/train/exp_augmented_glass2/weights/best.pt'
    volume_weights = 'volume_estimation/models/model_2024_11_24.pth'

    process_video(input_video, output_video, yolo_weights, volume_weights)
