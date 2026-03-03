"""Test CrossKD training with Knowledge Distillation."""
from ultralytics import YOLO

# Test CrossKD training (KD enabled)
print("Testing CrossKD training (KD=True)...")

# Load student model
model = YOLO("yolo11s.yaml")

# Train with CrossKD using yolov8s as teacher
results = model.train(
    data="coco-mini.yaml",
    epochs=600,
    batch=32,
    kd=True,  # Enable KD
    teacher_model="runs/detect/yolo11-l-baseline/weights/best.pt",  # Teacher model
    kd_temperature=1.0,
    kd_loss_weight_cls=1.0,
    kd_loss_weight_box=4.0,
    device=0,
    verbose=True,
    project='crosskd',
    name='teacher=L_student=S'
)
