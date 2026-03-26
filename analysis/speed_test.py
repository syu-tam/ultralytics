import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

model = YOLO('runs/detect/self_distill/yolo11l-bifpn-coco-mini/weights/best.onnx', task="detect")
#model = YOLO('yolo11m.onnx', task="detect")
image_dir = Path("/workspace/datasets/coco/images/val2017")
device = "cpu"
image_paths = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# warmup
for p in tqdm(image_paths[:10], desc="Warmup", unit="img"):
    _ = model.predict(source=str(p), imgsz=640, device=device, verbose=False, batch=1)

inf_ms = []
for p in tqdm(image_paths, desc="Benchmark", unit="img"):
    r = model.predict(source=str(p), imgsz=640, device=device, verbose=False, batch=1)[0]
    inf_ms.append(r.speed["inference"])

inf_ms = np.array(inf_ms, dtype=np.float64)
mean_ms = float(np.mean(inf_ms))
std_ms = float(np.std(inf_ms, ddof=1))  # 不偏標準偏差

print(f"n = {len(inf_ms)}")
print(f"inference mean = {mean_ms:.3f} ms")
print(f"inference std  = {std_ms:.3f} ms")
print(f"report style   = {mean_ms:.1f} ± {std_ms:.1f} ms")