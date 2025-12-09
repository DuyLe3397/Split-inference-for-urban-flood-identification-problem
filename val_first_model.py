from ultralytics import YOLO

model = YOLO("yolov8_version1.pt")

results = model.val(
    data="dataset/data.yaml",
    imgsz=640,
    batch=2,
    workers=0,
    device=0,
    save=True,
    save_json=True,
    project="runs",
    name="val_first_model",
    exist_ok=True
)

print("\n===== METRICS =====")
metrics = results.results_dict
for k, v in metrics.items():
    print(k, ":", v)
