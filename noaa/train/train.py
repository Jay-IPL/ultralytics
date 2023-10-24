from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/work/u1436961/JieMei/ultralytics/ultralytics/cfg/datasets/noaa_full.yaml',
                      epochs=50,
                      imgsz=640,
                      batch=32*8,
                      device=[0,1,2,3,4,5,6,7],
                      name='noaa_full(yolov8x)')
