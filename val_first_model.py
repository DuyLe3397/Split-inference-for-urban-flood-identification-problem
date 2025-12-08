from ultralytics import YOLO

model = YOLO("yolov8_version1.pt")

print("Bắt đầu chạy Validation...")

model.val(
    data="dataset/data.yaml",
    imgsz=640,
    batch=2,
    workers=0,
    device=0,
    save=True,
    save_json=True,

    project="runs",              # thư mục gốc
    name="val_first_model",      # thư mục con để lưu kết quả
    exist_ok=True               # ghi đè nếu đã tồn tại (tùy chọn)
)
