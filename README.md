# Tóm tắt cả hệ thống
Web (Client) sẽ bóc tách video đầu vào thành các frame còn nếu là ảnh thì chính là 1 frame, sau đó gửi từng frame về thông qua Fetch cho Server của Web lấy rồi gửi qua hàng đợi thông qua RabbitMQ, phía sau Client 1 của model AI thông qua RabbitMQ sẽ liên tục lên hàng đợi đó lấy từng frame về và chạy inference luôn rồi truyền cho Client 2 thông qua RabbitMQ rồi Client 3 thông qua RabbitMQ sẽ gửi dự đoán về theo từng mảng cho Server của model AI xử lý (mỗi frame sẽ tương ứng với 1 mảng chứa các dự đoán), sau đó trên Web (Client) sẽ gửi vị trí ghi nhận dữ liệu thông qua Fetch về cho Server của Web lấy rồi gửi lên hàng đợi thông qua RabbitMQ, Server của model AI thông qua RabbitMQ lên hàng đợi đó lấy về, sau đó xử lý dữ liệu dự đoán cùng dữ liệu tại vị trí đó, rồi thông qua RabbitMQ gửi lên hàng đợi để Server của Web thông qua RabbitMQ lấy về rồi gửi qua SSE cho Web (Client) lấy rồi hiển thị lên bản đồ.

## Model
![sl_model](pics/Model_Urban_Flood_Identification_Problem.jpg)

## Install the AI ​​model side packages
```
certifi            2025.11.12
matplotlib         3.10.7
numpy              1.26.4
opencv-python      4.8.1.78
pip                25.3 
PyYAML             6.0.3
requests           2.32.5 
torch              2.1.2+cu118  
torchvision        0.16.2+cu118 
ultralytics        8.0.196
ultralytics-thop   2.0.18 
urllib3            2.6.0 
```

## Install web-side packages
```
  amqplib       0.10.9
  body-parser   2.2.0
  cors          2.8.5
  dotenv        17.2.3
  ejs           3.1.10
  express       5.1.0
  multer        2.0.2
```

# split_inference

## Configuration
Application configuration is in the `config.yaml` file:
```yaml
name: YOLO
server:
  cut-layer: a #or b, c
  clients:
    - 1
    - 1
    - 1
  model: yolov8_version1 # trained model for urban flood identification problem
  batch-frame: 1
rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

data: video.mp4 
nc: 4 # number of class
log-path: .
control-count: 10
debug-mode: False
```

This configuration is use for server.

## How to Run
Alter your configuration, you need to run the server to listen and control the request from clients.
```commandline
python server.py
```

Now, when server is ready, run clients simultaneously with total number of client that you defined.


**Layer 1**

```commandline
python client.py --layer_id 1 
```


**Layer 2**

```commandline
python client.py --layer_id 2 
```


**Layer 3**

```commandline
python client.py --layer_id 3 
```
In the computer, execute the command `cd UrbanFloodMonitoringApplication` and then run the following command to start the web.

**Web**

```commandline
node server.js 
```


Where:
- `--layer_id` is the ID index of client's layer, start from 1. 


If you want to use a specific device configuration for the training process, declare it with the `--device` argument when running the command line:

```commandline
python client.py --layer_id 1 --device cpu
```