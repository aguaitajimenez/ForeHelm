import os
from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO('yolo11n.pt')  # Choose the appropriate model variant

# Train the model
path = os.getcwd()
yaml_path = os.path.join(path, "config.yaml")
model.train(
    data=yaml_path,  # Path to the data configuration file
    epochs=25,         # Number of training epochs
    imgsz=640,         # Image size
    batch=16,          # Batch size
    device="cpu"           # GPU device (use 'cpu' for CPU training)
)