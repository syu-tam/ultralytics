from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("best.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="datasets/coco.yaml", 
                      batch=64,
                      epochs=600,
                      optimizer='SGD',
                      scale=0.9,
                      mixup=0.15,
                      copy_paste=0.4)
