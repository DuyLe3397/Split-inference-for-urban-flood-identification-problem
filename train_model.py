# server.py
import json
import base64
import io
import numpy as np
import cv2
import pika
from ultralytics import YOLO

RABBIT_URL = 'amqp://guest:guest@localhost:5672/%2F'  # thay chuỗi kết nối phù hợp
QUEUE_NAME = 'image_queue'

# Tải model YOLOv8 (vd: yolov8n.pt). Đặt file .pt cùng thư mục hoặc chỉ định đường dẫn.
model = YOLO('yolov8n.pt')  # hoặc 'yolov8s.pt' tùy bạn


def decode_image_base64_to_cv2(image_b64: str):
    """Giải mã base64 -> OpenCV image (BGR)."""
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    return img


def on_message(ch, method, properties, body):
    try:
        payload = json.loads(body.decode('utf-8'))
        print(
            f"[x] Nhận message id={payload.get('id')} file={payload.get('filename')}")

        lat = payload.get('lat')
        lon = payload.get('lon')
        image_b64 = payload.get('image_base64')

        # Giải mã ảnh
        img_bgr = decode_image_base64_to_cv2(image_b64)
        if img_bgr is None:
            print("Không decode được ảnh.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # YOLOv8 inference
        results = model(img_bgr, verbose=False)
        # Lấy kết quả dạng đơn giản: boxes với label + conf
        detections = []
        for r in results:
            boxes = r.boxes
            names = r.names  # map id->label
            for b in boxes:
                cls_id = int(b.cls[0])
                label = names.get(cls_id, str(cls_id))
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                detections.append({
                    "label": label,
                    "conf": round(conf, 3),
                    "xyxy": [round(v, 1) for v in xyxy]
                })

        print(f"Vị trí: lat={lat}, lon={lon}")
        print(f"Phát hiện: {detections}")

        # TODO: Bạn có thể:
        # - Lưu ảnh kèm bbox (cv2.rectangle) -> file
        # - Publish kết quả sang queue khác (ví dụ: result_queue)
        # - Lưu log DB

        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print("Lỗi xử lý message:", e)
        # Tùy chính sách, ack hoặc nack/requeue
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    params = pika.URLParameters(RABBIT_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message)
    print(f"[RabbitMQ] Đang lắng nghe queue '{QUEUE_NAME}' ...")
    channel.start_consuming()


if __name__ == "__main__":
    main()
