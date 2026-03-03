"""Test baseline training without Knowledge Distillation."""
from ultralytics import YOLO

# Test baseline training (KD disabled)
print("Testing baseline training (KD=False)...")
model = YOLO("yolo11l.yaml")

# Train for 1 epoch to verify everything works
results = model.train(
    data="coco-mini.yaml",
    epochs=600,
#     imgsz=64,
     batch=32,
     kd=False,  # Explicitly disable KD
#     device=0,
#     verbose=True,
    optimizer='SGD',
    momentum=0.9
)
