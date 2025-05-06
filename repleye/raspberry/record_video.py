from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time
from datetime import datetime

def record_video():
    # Initialize the camera
    picam2 = Picamera2()
    
    # Configure with full raw sensor size and 1920x1080 output
    picam2.configure(picam2.create_video_configuration(
        raw={"size": (2592, 1944)},  # Full sensor size
        main={"size": (1920, 1080)}  # Output at 1920x1080
    ))
    
    # Generate filename with current date and number
    filename = f"train_{datetime.now().strftime('%d_%m_%Y')}_full_2.h264"
    
    print(f"Starting video recording for 25 seconds...")
    print(f"Video will be saved as: {filename}")
    print(f"Recording at 1920x1080 resolution (from full sensor)")
    
    # Create encoder and start recording
    encoder = H264Encoder(bitrate=20000000)  # 20Mbps for high quality
    picam2.start_recording(encoder, filename)
    
    # Record for 25 seconds
    time.sleep(25)
    
    # Stop recording
    picam2.stop_recording()
    print("Recording completed!")
    
    # Close the camera
    picam2.close()

if __name__ == "__main__":
    record_video() 