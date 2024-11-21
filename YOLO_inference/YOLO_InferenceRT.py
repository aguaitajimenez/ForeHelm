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

    # Load YOLO model ncnn format
    # model_name = "trainin7g0_25epochs.pt"
    model_name = "yolo11n_ncnn_model"
    # model_name = "yolo11n.pt"
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
        result = model.predict(frame, save=False, conf=0.5)[0]
        out_frame =  result.plot()

        # Calculate FPS
        current_time = time.time()
        inference_period = result.speed["inference"]
        period = (current_time - prev_time)
        fps = 1 / period
        prev_time = current_time

        # Display FPS on the result (bottom-left corner)
        fps_label = f"FPS: {fps:.2f}"
        period_label = f"Period: {period*1000:.0f} ms"
        inference_label = f"Inference: {inference_period:.0f} ms"
        
        cv2.putText(out_frame, inference_label, (10, out_frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out_frame, period_label, (10, out_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out_frame, fps_label, (10, out_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        
        # Display result
        cv2.imshow(model_name+" Live Detection", out_frame)

        # Exit on pressing 'ESC'
        if cv2.waitKey(30) == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
