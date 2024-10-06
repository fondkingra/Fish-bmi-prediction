import cv2
import numpy as np
from ultralytics import YOLO

def calculate_fish_metrics(bbox, scale_factor):
    x, y, width, height = bbox
    height_cm = height * scale_factor
    width_cm = width * scale_factor
    bmi = width_cm / (height_cm ** 2)
    return height_cm, width_cm, bmi

def main(input_video_path, output_video_path, scale_factor=0.5):
    # Load your trained YOLOv8 model
    model_path = r'E:\fish bmi\fishes trained model.pt'
    model = YOLO(model_path)  # Load YOLOv8 model

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)  # Run inference on the frame

        # Extract bounding boxes
        for bbox in results[0].boxes.xyxy:  # Adjust for YOLOv8 results structure
            x1, y1, x2, y2 = map(int, bbox[:4])
            width = x2 - x1
            height = y2 - y1
            
            # Calculate metrics
            height_cm, width_cm, bmi = calculate_fish_metrics((x1, y1, width, height), scale_factor)

            # Annotate frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f'Height: {height_cm:.2f} cm', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Width: {width_cm:.2f} cm', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'BMI: {bmi:.2f}', (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the annotated frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video = r'E:\fish bmi\fishes video.mp4'  # Input video path
output_video = r'E:\fish bmi\output.mp4'  # Output video path
main(input_video, output_video)
