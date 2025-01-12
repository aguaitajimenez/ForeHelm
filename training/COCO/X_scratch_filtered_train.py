from ultralytics import YOLO

modelX = YOLO("yolo11n.yaml")
modelX.train(
    data="train_filtered.yaml",
    project="./filtered_from_scratch_X",
    pretrained=False,  
    epochs = 100, 
    patience=10, 
    save_period=10,
    batch=-1,
    resume=True, 
    save=True
    )