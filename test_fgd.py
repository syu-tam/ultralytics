"""Smoke test for FGD knowledge distillation.

Run inside Docker container:
  docker exec -it yolo-dev bash
  cd /workspace/ultralytics
  python test_fgd.py

Verification checklist:
  1. Loss log shows 7 items: box, cls, dfl, fgd_fg, fgd_bg, fgd_mask, fgd_rela
  2. kd_type="crosskd" still works (regression check)
  3. kd=False shows 3-item loss (standard training)
  4. All 3 epochs complete with finite losses (no NaN)
"""

import sys

TEACHER_MODEL = "yolo11s.pt"  # change to actual teacher checkpoint path
DATA = "coco8.yaml"

from ultralytics import YOLO


def test_fgd():
    print("\n=== Test 1: FGD training (kd_type='fgd') ===")
    model = YOLO("yolo11n.yaml")
    model.train(
        data=DATA,
        epochs=1,
        batch=2,
        kd=True,
        kd_type="fgd",
        teacher_model=TEACHER_MODEL,
        fgd_temp=0.5,
        fgd_alpha=0.0016,
        fgd_beta=0.0008,
        fgd_gamma=0.0008,
        fgd_lambda=0.000008,
        kd_freeze_teacher=True,
        device=0,
        project="fgd",
        name="yolo11s-teacher_yolo11l",
        verbose=True,
    )
    print("Test 1 PASSED")


def test_crosskd_regression():
    print("\n=== Test 2: CrossKD regression (kd_type='crosskd') ===")
    model = YOLO("yolo11n.yaml")
    model.train(
        data=DATA,
        epochs=3,
        batch=2,
        kd=True,
        kd_type="crosskd",
        teacher_model=TEACHER_MODEL,
        kd_freeze_teacher=True,
        device=0,
        project="fgd",
        name="crosskd-regression",
        verbose=True,
    )
    print("Test 2 PASSED")


def test_standard():
    print("\n=== Test 3: Standard training (kd=False) ===")
    model = YOLO("yolo11n.yaml")
    model.train(
        data=DATA,
        epochs=3,
        batch=2,
        kd=False,
        device=0,
        project="fgd",
        name="standard-baseline",
        verbose=True,
    )
    print("Test 3 PASSED")


if __name__ == "__main__":
    tests = sys.argv[1:] or ["fgd", "crosskd", "standard"]
    if "fgd" in tests:
        test_fgd()
    if "crosskd" in tests:
        test_crosskd_regression()
    if "standard" in tests:
        test_standard()
    print("\nAll requested tests PASSED")
