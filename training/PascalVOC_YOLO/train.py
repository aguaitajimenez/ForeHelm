# !pip install ultralytics 
from ultralytics import YOLO
import os

epochs_per_run = 5

path = os.getcwd()
model_path = os.path.join(path, "runs/detect/successful_train_25", "weights/last.pt")

i=0

print("*********************")
print("*********************")
print(f'***Saved epoch {i}***')
print("*********************")
print("*********************")

while 1:

    # Load a pretrained YOLOv11 model
    model = YOLO(model_path)  # Choose the appropriate model variant
    
    # Train the model
    yaml_path = os.path.join(path, "train.yaml")
    print(yaml_path)
    out = model.train(
        data=yaml_path,           # Path to the data configuration file
        epochs=epochs_per_run,    # Number of training epochs
        imgsz=640,                # Image size
        batch=16,                 # Batch size
        device="cpu"              # GPU device (use 'cpu' for CPU training)
    )

    model_path = os.path.join(path, out.save_dir, "weights/best.pt")
    i+=epochs_per_run
    print("*********************")
    print("*********************")
    print(f'***Saved epoch {i}***')
    print("*********************")
    print("*********************")