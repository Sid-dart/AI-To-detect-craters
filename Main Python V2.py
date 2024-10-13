import torch
from ultralytics import YOLO

def main():
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the pre-trained YOLOv8 model weights

    # Train the model
    model.train(
        data='Main yml.yaml',  # Path to the data configuration file
        epochs=100,        # Number of training epochs
        batch=3,          # Batch size (correct argument)
        imgsz=416,         # Image size set to 416x416
        device=device,     # Device to use
        project='runs/train',  # Project name for saving the training results
        name='exp'             # Experiment nam
         )

if __name__ == '__main__':
    main()