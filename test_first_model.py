import os
from glob import glob
from ultralytics import YOLO
import cv2


def main():
    # === Load model YOLOv8l với weights best.pt ===
    weights = "yolov8_version1.pt"   # File đã train bằng yolov8l
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Không tìm thấy weights: {weights}")

    # nạp mô hình
    model = YOLO(weights)   # tự nhận đúng loại: yolov8l

    # === Folder chứa ảnh test ===
    print("Bắt đầu test ảnh trong dataset")
    image_files = sorted([
        f for f in os.listdir('dataset/test/images')
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(image_files)} images")
    for img_name in image_files:
        img_path = os.path.join('dataset/test/images', img_name)
        # Load ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Không đọc được ảnh: {img_path}")
            continue
        print(f"Đã đọc được tên ảnh: {img_name}")
        # === Predict ===
        model.predict(
            source=img_path,
            imgsz=640,
            conf=0.182,
            iou=0.45,
            device=0 if model.device.type == "cuda" else "cpu",
            save=True,
            save_txt=True,
            save_conf=True,
            project="runs",
            name="test_fisrt_model",
            exist_ok=True,
            verbose=True
        )


if __name__ == "__main__":
    main()
