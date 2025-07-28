from ultralytics import YOLO

# Load model (use pre-trained weights or train from scratch)
model = YOLO("yolov11n.pt")  # Choose a suitable model

if __name__ == "__main__":
    model.train(
        data=r"C:\Users\muham\Documents\Caries_detection(3)\data.yaml",  # Path to dataset config
        epochs=50,
        batch=16,
        imgsz=640,
        device="cuda"  # Use 'cpu' if no GPU
    )