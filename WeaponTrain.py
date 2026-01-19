from ultralytics import YOLO

# Load a pre-trained YOLOv8 model for fine-tuning
model = YOLO("yolov8n.pt")  # Use a smaller or larger model based on need

# Train on your dataset (update the dataset path)
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="D:\\ProjectVision\\Weapon-2-2\\data.yaml", epochs=50, imgsz=640)
    # Save the trained model
    print("Done")