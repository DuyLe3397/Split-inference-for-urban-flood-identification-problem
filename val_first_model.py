from ultralytics import YOLO
import psutil
import os
import time

# ====== Hàm đo RAM ======
process = psutil.Process(os.getpid())


def ram_mb():
    return process.memory_info().rss / 1024 / 1024


ram1 = ram_mb()

# ====== Load model ======
model = YOLO("yolov8_version1.pt")
ram2 = ram_mb()

# ====== Chạy validation và đo peak RAM ======
ram_log = []

start_time = time.time()

results = model.val(
    data="dataset/data.yaml",
    imgsz=640,
    batch=2,
    workers=0,
    device=0,        # GPU nếu có
    save=True,
    save_json=True,
    project="runs",
    name="val_first_model",
    exist_ok=True
)

# Lấy RAM sau khi val
ram3 = ram_mb()

# ====== In metrics ======
print("\n===== METRICS =====")
metrics = results.results_dict
for k, v in metrics.items():
    print(k, ":", v)

# ====== Thống kê RAM ======
end_time = time.time()

print("\n===== RAM USAGE =====")
print(f"RAM ban đầu:           {ram1:.2f} MB")
print(f"RAM sau load model:    {ram2:.2f} MB")
print(f"RAM sau khi chạy val:  {ram3:.2f} MB")
print(f"Thời gian chạy val:    {end_time - start_time:.2f} giây")
