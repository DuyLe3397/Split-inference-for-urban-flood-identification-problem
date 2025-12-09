from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Đọc tên ảnh mà server vừa dùng
with open("raw_split_info.txt", "r", encoding="utf-8") as f:
    img_name = f.read().strip()

img_path = f"dataset/valid/images/{img_name}"
print("Using image:", img_path)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# TODO: nếu client có resize/pad khác 640x640 thì bạn cần copy đúng y chang;
# demo dưới đây dùng resize 640x640 đơn giản:
img_resized = cv2.resize(img, (640, 640))
im = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
im = im.unsqueeze(0)  # [1, 3, 640, 640]

model = YOLO("yolov8_version1.pt")
with torch.no_grad():
    raw_full = model.model(im)[0]  # tùy version: có thể là model(im)[0]
    print("raw_full shape:", raw_full.shape)
    torch.save(raw_full.cpu(), "raw_full_model.pt")
