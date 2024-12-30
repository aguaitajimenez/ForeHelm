from ultralytics import YOLO

model16 = YOLO("yolo11n.yaml")
model16.train(
    data="train_filtered.yaml",
    project="./filtered_from_scratch_16",
    pretrained=False,  
    epochs = 100, 
    patience=10, 
    save_period=10,
    batch=16,
    resume=True, 
    save=True
    )