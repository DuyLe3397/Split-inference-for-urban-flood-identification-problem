from ultralytics import YOLO

print("Đang load model: best_85epochs.pt")
model = YOLO("best_85epochs.pt")

print("Bắt đầu chạy Validation...")
model.val(
    data="cacacacaca-3/data.yaml",
    imgsz=640,
    batch=2,
    workers=0,        # BẮT BUỘC CHO WINDOWS
    device=0,         # dùng GPU
    save=True,
    save_json=True,
)
