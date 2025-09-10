# Install ultralytics if not already installed
# pip install ultralytics

from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (you can choose n/s/m/l depending on speed/accuracy needs)
model = YOLO("yolov8n.pt")  # 'n' = nano (fastest, smallest), 's' = small, 'm' = medium, 'l' = large

# Train the model
results = model.train(
    data="dataset/data.yaml",  # Path to dataset YAML (we’ll create this below)
    epochs=50,                 # Number of training epochs
    imgsz=640,                 # Image size (resize for training)
    batch=2,                  # Batch size
    workers=4,                 # Number of dataloader workers
    device="cpu",              # GPU (0) or CPU ('cpu')
    name="plate-detector"      # Experiment name
)
