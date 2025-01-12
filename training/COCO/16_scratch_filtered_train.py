from ultralytics import YOLO
import sys

model16 = YOLO("yolo11n.yaml")
model16.train(
    data="train_filtered.yaml",
    project="./16_scratch_filtered",
    pretrained=False,  
    epochs = 100,
    patience=10, 
    save_period=5,
    # time=2,
    # cache="ram",
    batch=16,
    exist_ok=True,
    resume=True, 
    plots=True
    )