from ultralytics import YOLO

# Load a pre-trained YOLOv11 detection model
model = YOLO("your_custom_model.pt")

# Export the model to ONNX format
model.export(format="onnx")  

# Print the model's class names
print(model.names)
