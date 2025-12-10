import os
from ultralytics import YOLO
from roboflow import Roboflow


def main():
    # ============================
    # 1. TẢI DATASET TỪ ROBOFLOW
    # ============================
    rf = Roboflow(api_key="d6Ftxh41wdgfXKN8QeoO")
    project = rf.workspace("duy-6d40k").project("cacacacaca-smvlx")
    version = project.version(3)

    print("Đang tải dataset từ Roboflow...")
    # dataset.location là đường dẫn dataset
    dataset = version.download("yolov8")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError("Không tìm thấy file data.yaml trong dataset!")

    # ============================
    # 2. CHỌN MODEL YOLOv8
    # ============================
    model_name = "yolov8n.pt"  # hoặc yolov8s.pt, yolov8m.pt, yolov8l.pt
    print(f"Đang load model {model_name}...")
    model = YOLO(model_name)

    # ============================
    # 3. TRAIN MODEL
    # ============================
    print("Bắt đầu train YOLOv8...")

    model.train(
        data=data_yaml,
        epochs=40,               # ---- đổi số vòng lặp ở đây
        imgsz=640,
        batch=4,
        workers=0,
        device=0,                 # GPU 0 (nếu không có GPU thì tự chuyển CPU)
        project="runs",
        name="train_yolov8",
        exist_ok=True,
        verbose=True,
        plots=True
    )

    print("\n🎉 TRAIN HOÀN THÀNH! Weights lưu tại: runs/train_yolov8/weights/best.pt\n")


if __name__ == "__main__":
    main()
