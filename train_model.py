import os
from ultralytics import YOLO

# code này sẽ dược train trên colab vì có GPU cuda nên sẽ train nhanh hơn, sau khi có file tham số thì lấy về đưa vào dự án


def main():
    # Cấu hình căn bản
    data_yaml = "data.yaml"  # cập nhật đường dẫn nếu để nơi khác
    # Chọn kiến trúc:
    # - Detect: "yolov8n.pt", "yolov8s.pt", ...
    # - Segment: "yolov8n-seg.pt", "yolov8s-seg.pt", ...
    model_name = "yolov8n.pt"  # đổi thành "yolov8n-seg.pt" nếu bạn muốn segmentation
    # Tạo/Load model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=100,            # chỉnh theo dataset
        imgsz=640,             # kích thước ảnh train
        batch=16,              # chỉnh theo VRAM
        device=0,          # GPU id; dùng "cpu" nếu không có GPU
        workers=4,             # số luồng dataloader
        optimizer="auto",      # để auto chọn; hoặc "SGD"/"AdamW"
        lr0=0.01,              # lr khởi đầu (tuỳ chỉnh)
        weight_decay=0.0005,
        patience=50,           # early stopping
        project="runs",        # thư mục gốc
        name="train",          # tên run
        exist_ok=True,         # ghi đè nếu tồn tại
        verbose=True
    )

    # Val sau train (Ultralytics sẽ tự val trong quá trình train; đây là val riêng nếu muốn)
    metrics = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        device="cpu"
    )
    print("Validation metrics:", metrics)

    # Đường dẫn weights
    # best.pt được lưu tại: runs/detect/train/weights/best.pt (task detect)
    # hoặc: runs/segment/train/weights/best.pt (task segment)
    weights_dir = model.trainer.best if hasattr(model, "trainer") else None
    print("Best weights saved at:", weights_dir)


main()

# import os
# from ultralytics import YOLO
# from roboflow import Roboflow


# def main():
#     # ============================
#     # 1. TẢI DATASET TỪ ROBOFLOW
#     # ============================
#     rf = Roboflow(api_key="d6Ftxh41wdgfXKN8QeoO")
#     project = rf.workspace("duy-6d40k").project("cacacacaca-smvlx")
#     version = project.version(3)

#     print("Đang tải dataset từ Roboflow...")
#     # dataset.location là đường dẫn dataset
#     dataset = version.download("yolov8")

#     data_yaml = os.path.join(dataset.location, "data.yaml")
#     if not os.path.exists(data_yaml):
#         raise FileNotFoundError("Không tìm thấy file data.yaml trong dataset!")

#     # ============================
#     # 2. CHỌN MODEL YOLOv8
#     # ============================
#     model_name = "yolov8n.pt"  # hoặc yolov8s.pt, yolov8m.pt, yolov8l.pt
#     print(f"Đang load model {model_name}...")
#     model = YOLO(model_name)

#     # ============================
#     # 3. TRAIN MODEL
#     # ============================
#     print("Bắt đầu train YOLOv8...")

#     model.train(
#         data=data_yaml,
#         epochs=41,               # ---- đổi số vòng lặp ở đây
#         imgsz=640,
#         batch=4,
#         workers=0,
#         device=0,                 # GPU 0 (nếu không có GPU thì tự chuyển CPU)
#         project="runs",
#         name="train_yolov8",
#         exist_ok=True,
#         verbose=True,
#         plots=True
#     )

#     print("\n🎉 TRAIN HOÀN THÀNH! Weights lưu tại: runs/train_yolov8/weights/best.pt\n")


# if __name__ == "__main__":
#     main()
