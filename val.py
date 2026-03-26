from ultralytics import YOLO

model = YOLO('runs/detect/self_distill/yolo11l-bifpn-coco-mini/weights/best.pt')  # モデルのパス
#model = YOLO('yolo11s.pt')
results = model.val(data='coco128.yaml', split='val',save=False, batch=1, imgsz=640, device=0)



