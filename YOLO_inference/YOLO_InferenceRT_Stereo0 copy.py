import cv2
from ultralytics import YOLO
import threading
import queue
import argparse
import time  # Import time for FPS calculation

# Shared flag to signal threads to stop
stop_flag = threading.Event()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 multi-camera live detection")
    parser.add_argument(
        "--webcam-resolution", 
        default=[640, 420],
        nargs=2, 
        type=int,
        help="Resolution for the webcams in [width height] format."
    )
    args = parser.parse_args()
    return args

def camera_thread(camera_id, frame_width, frame_height, frame_queue):
    """
    Thread function for processing a single camera.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}.")
        return

    # Load YOLO model (individual instance per thread)
    model = YOLO("yolo11n.pt")
    class_names = model.names  # Access class names from the model
    prev_time = time.time()  # For FPS calculation

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_id}.")
            break

        # Perform YOLO detection
        result = model.predict(frame, save=False, conf=0.5)[0]
        out_frame =  result.plot()

        # Calculate FPS
        current_time = time.time()
        inference_period = result.speed["inference"]
        period = (current_time - prev_time)
        fps = 1 / period
        prev_time = current_time

        # Display FPS on the frame
        fps_label = f"FPS: {fps:.2f}"
        period_label = f"Period: {period*1000:.0f} ms"
        inference_label = f"Inference: {inference_period:.0f} ms"
        
        cv2.putText(out_frame, inference_label, (10, out_frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out_frame, period_label, (10, out_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out_frame, fps_label, (10, out_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # Add the frame to the queue
        if not frame_queue.full():
            frame_queue.put((camera_id, out_frame))

    # Cleanup
    cap.release()

def main():
    global stop_flag
    # Parse arguments
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Camera IDs (change based on your setup)
    camera_ids = [0, 2]  # Adjust these for your cameras

    # Queue to hold frames
    frame_queue = queue.Queue(maxsize=10)

    # Create and start threads for each camera
    threads = []
    for cam_id in camera_ids:
        thread = threading.Thread(target=camera_thread, args=(cam_id, frame_width, frame_height, frame_queue))
        thread.start()
        threads.append(thread)

    print("Press 'q' to exit the program.")

    # Main thread for displaying frames
    while not stop_flag.is_set():
        while not frame_queue.empty():
            camera_id, frame = frame_queue.get()
            cv2.imshow(f"YOLOv8 Live Detection - Camera {camera_id}", frame)

        # Check for 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
