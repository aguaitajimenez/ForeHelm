import cv2
from ultralytics import YOLO
import argparse
import time  # Import time for FPS calculation

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live detection")
    parser.add_argument(
        "--webcam-resolution", 
        default=[640, 480],
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLO model
    # model_name = "trainin7g0_25epochs.pt"
    model_name = "yolov8n"
    model = YOLO(model_name)
    class_names = model.names  # Access class names from the model

    # Variables for FPS calculation
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Perform prediction
        results = model(frame, conf=0.5)[0]  # Get the first (and only) result

        # Process detections
        for box in results.boxes:
            # Extract box coordinates, confidence, and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = class_names[class_id]  # Get the class name

            # Draw the box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"  # Use class name
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS on the frame (bottom-left corner)
        fps_label = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_label, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display frame
        cv2.imshow(model_name+" Live Detection", frame)

        # Exit on pressing 'ESC'
        if cv2.waitKey(30) == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
