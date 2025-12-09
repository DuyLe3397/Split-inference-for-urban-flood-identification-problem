# Test.py - Vẽ bbox từ YOLO predictions và lưu ảnh kết quả

import os
import cv2
import numpy as np


class Tester:
    def __init__(self,
                 img_folder='dataset/test/images',
                 save_folder='runs/test_splited_model'):
        """
        img_folder : thư mục chứa ảnh gốc (nơi tìm img_name)
        save_folder: thư mục lưu ảnh đã vẽ bbox
        """
        self.img_folder = img_folder
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    @staticmethod
    def scale_pred_to_pixel(pred, orig_shape):
        """
        Chuyển 1 prediction từ YOLO normalized sang pixel xyxy.

        pred: [cls, cx, cy, w, h, conf]
            - cls : class id (int hoặc float)
            - cx, cy, w, h: normalized (0..1) theo (W, H)
            - conf: confidence
        orig_shape: (H, W) của ảnh gốc
        """
        H, W = orig_shape
        cls = int(pred[0])
        cx = float(pred[1]) * W
        cy = float(pred[2]) * H
        bw = float(pred[3]) * W
        bh = float(pred[4]) * H
        conf = float(pred[5])

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        box = np.array([x1, y1, x2, y2], dtype=float)
        return cls, conf, box

    def __call__(self, yolo_preds, orig_shape, img_name):
        """
        Hàm chính để vẽ bbox cho 1 ảnh.

        yolo_preds: list các bbox YOLO normalized
                    mỗi phần tử: [cls, cx, cy, w, h, conf]
        orig_shape: (H, W) kích thước ảnh gốc
        img_name  : tên file ảnh (ví dụ: "0001.jpg")
        """
        # Đọc ảnh gốc
        img_path = os.path.join(self.img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Tester] Cannot read image: {img_path}")
            return

        # Nếu orig_shape khác với img.shape, có thể resize cho khớp (nếu cần)
        # Ở đây giả định orig_shape khớp với ảnh trong img_folder.
        H, W = orig_shape
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H))

        # Vẽ từng bbox
        for pred in yolo_preds:
            cls, conf, box = self.scale_pred_to_pixel(pred, orig_shape)
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to image bounds cho an toàn
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            # Vẽ bbox (màu đỏ)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Vẽ text: class:conf
            label = f"{cls}:{conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Lưu ảnh kết quả
        save_path = os.path.join(self.save_folder, img_name)
        cv2.imwrite(save_path, img)
        print(f"[Tester] Saved result to {save_path}")
