import os
from glob import glob
from ultralytics import YOLO


def main():
    # === Load model YOLOv8l với weights best.pt ===
    weights = "yolov8_version1.pt"   # File đã train bằng yolov8l
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Không tìm thấy weights: {weights}")

    # nạp mô hình
    model = YOLO(weights)   # tự nhận đúng loại: yolov8l

    # === Folder chứa ảnh test ===
    source_dir = "dataset/test/images"
    image_paths = sorted(glob(os.path.join(source_dir, "*.*")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"Không thấy ảnh trong thư mục: {source_dir}")

    # === Predict ===
    results = model.predict(
        source=source_dir,
        imgsz=640,
        conf=0.182,
        iou=0.45,
        device=0 if model.device.type == "cuda" else "cpu",
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs",
        name="predict_fisrt_model",
        exist_ok=True,
        verbose=True
    )

    # === In kết quả 5 ảnh đầu ===
    for i, r in enumerate(results[:5]):
        print(
            f"[IMAGE {i}] shape={r.orig_shape}, detections={len(r.boxes)}, "
            f"boxes={r.boxes.data.cpu().numpy()}"
        )


if __name__ == "__main__":
    main()
