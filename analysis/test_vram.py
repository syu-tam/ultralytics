"""VRAM measurement script for Baseline and Ours (self-distill) across model sizes.

Usage (inside Docker):
    docker exec yolo-dev bash -c "cd /workspace/ultralytics && python test_vram.py"

Results are saved to vram_results.csv.
"""

import csv
from pathlib import Path

import torch
from ultralytics import YOLO

DATA = "coco-mini.yaml"
DEVICE = 0
OUTPUT = Path("vram_results.csv")


def measure(label, yaml, batch, **train_kwargs):
    torch.cuda.empty_cache()

    model = YOLO(yaml)
    torch.cuda.reset_peak_memory_stats()
    model.train(
        data=DATA,
        epochs=2,
        batch=batch,
        device=DEVICE,
        plots=False,
        verbose=False,
        **train_kwargs,
    )

    alloc    = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved()  / 2**30
    print(f"{label:20s}: allocated={alloc:.2f}G  reserved={reserved:.2f}G")
    return alloc, reserved


_KD = {
    "crosskd": dict(kd=True, kd_type="crosskd", teacher_model="yolo11l-teacher.pt"),
    "fgd":     dict(kd=True, kd_type="fgd",     teacher_model="yolo11l-teacher.pt"),
}

CASES = [
    ("S-Baseline",    "yolo11s.yaml",                    64, {}),
    ("S-CrossKD",     "yolo11s.yaml",                    64, _KD["crosskd"]),
    ("S-FGD",         "yolo11s.yaml",                    64, _KD["fgd"]),
    ("S-Ours",        "yolo11s-self-distill.yaml",        64, {}),
]

if __name__ == "__main__":
    print(f"{'label':20s}  {'allocated':>12s}  {'reserved':>12s}")
    print("-" * 50)

    rows = []
    for label, yaml, batch, kwargs in CASES:
        alloc, reserved = measure(label, yaml, batch, **kwargs)
        rows.append({"label": label, "allocated_G": f"{alloc:.2f}", "reserved_G": f"{reserved:.2f}"})

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "allocated_G", "reserved_G"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {OUTPUT}")
