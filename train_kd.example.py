"""Training scripts for Knowledge Distillation methods.

Usage (inside Docker container):
  python train_kd.py [baseline|crosskd|fgd|self_distill|self_distill_fpn|self_distill_bifpn]

  # multiple methods
  python train_kd.py crosskd fgd

  # all methods
  python train_kd.py
"""

import sys

TEACHER_MODEL = "yolo11l-teacher.pt"
DATA = "coco-mini.yaml"
DEVICE = 0
EPOCHS = 600
BATCH = 32


def train_baseline():
    from ultralytics import YOLO

    model = YOLO("yolo11s.yaml")
    model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        kd=False,
        device=DEVICE,
        optimizer="SGD",
        momentum=0.9,
        project="baseline",
        name="yolo11s",
    )


def train_crosskd():
    from ultralytics import YOLO

    model = YOLO("yolo11s.yaml")
    model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        kd=True,
        kd_type="crosskd",
        teacher_model=TEACHER_MODEL,
        kd_freeze_teacher=True,
        kd_loss_weight_cls=1.0,
        kd_loss_weight_box=4.0,
        kd_temperature=1.0,
        device=DEVICE,
        optimizer="SGD",
        momentum=0.9,
        project="crosskd",
        name="yolo11s-teacher_yolo11l",
    )


def train_fgd():
    from ultralytics import YOLO

    model = YOLO("yolo11s.yaml")
    model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        kd=True,
        kd_type="fgd",
        teacher_model=TEACHER_MODEL,
        kd_freeze_teacher=True,
        fgd_temp=0.5,
        fgd_alpha=0.0001,
        fgd_beta=0.002,
        fgd_gamma=0.0005,
        fgd_lambda=0.000001,
        device=DEVICE,
        optimizer="SGD",
        momentum=0.9,
        project="fgd",
        name="yolo11s-teacher_yolo11l",
    )


def train_self_distill(aux_neck="pan"):
    from ultralytics import YOLO

    _YAML = {
        "pan": "yolo11s-self-distill.yaml",
        "fpn": "yolo11s-self-distill-fpn.yaml",
        "bifpn": "yolo11s-self-distill-bifpn.yaml",
    }
    model = YOLO(_YAML[aux_neck])
    model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        sd_temp_cls=0.5,
        sd_temp_dfl=0.5,
        sd_kd_weight=1.0,
        device=DEVICE,
        optimizer="SGD",
        momentum=0.9,
        project="self_distill",
        name=f"yolo11s-{aux_neck}",
    )


_METHODS = {
    "baseline": train_baseline,
    "crosskd": train_crosskd,
    "fgd": train_fgd,
    "self_distill": train_self_distill,
    "self_distill_fpn": lambda: train_self_distill("fpn"),
    "self_distill_bifpn": lambda: train_self_distill("bifpn"),
}

if __name__ == "__main__":
    requested = sys.argv[1:] or list(_METHODS.keys())
    unknown = [m for m in requested if m not in _METHODS]
    if unknown:
        print(f"Unknown methods: {unknown}. Available: {list(_METHODS.keys())}")
        sys.exit(1)
    for name in requested:
        _METHODS[name]()
