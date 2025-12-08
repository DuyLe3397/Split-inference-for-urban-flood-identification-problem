import pika
import uuid
import argparse
import yaml

import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

# tạo 1 đối tượng parser là đối tượng để xử lý các đối số nhập
parser = argparse.ArgumentParser(description="Split learning framework")
# từ dòng lệnh terminal, phần description sẽ hiển thị khi bấm help

# gõ từ terminal có tham số --layer_id phải cách ra rồi thêm số nguyên bắt đầu từ 1, và bắt buộc phải cung cấp, help sẽ hiển thị giải thích
# gõ từ terminal phải có optional --, còn thành phần trong parser là layer_id, cũng chính là layer để chia model
parser.add_argument('--layer_id', type=int, required=True,
                    help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False,
                    help='Device of client')
# tương tự trên nhưng --device cách ra và thêm string như cuda

# sẽ có kết quả gồm layer_id là số đã nhập, device là tên thiết bị đã thêm
args = parser.parse_args()

file = open('config.yaml', encoding="utf8")
config = yaml.safe_load(file)

# with open('config.yaml', 'r') as file: # chính là từ file cấu hình tự định nghĩa, bây giờ nó là file
#     config = yaml.safe_load(file) # tải file đó rồi lưu vào config

# lấy id ngẫu nhiên độc nhất và không bị trùng với bất cứ cái nào, nó sẽ làm id cho client
client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

device = None  # lấy device

if args.device is None:  # kiểm tra trong lệnh nhập từ terminal có nhập device không, nếu không thì
    if torch.cuda.is_available():  # kiểm tra nếu có cuda thì lưu thành cuda
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
        src.Log.Logger(f"result.log").log_info(
            f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"  # nếu không có cuda thì dùng cpu
        print(f"Using device: CPU")
        src.Log.Logger(f"result.log").log_info("Using device: CPU")
else:  # kiểm tra nếu lệnh nhập có thêm device
    device = args.device  # thì thêm device vào
    print(f"Using device: {device}")

# sử dụng thư viện pika để kết nối đến RabbitMQ
credentials = pika.PlainCredentials(username, password)
# ở đây tạo ra 1 giấy chứng nhận hay 1 cái giấy phép đơn giản từ tên và mật khẩu
connection = pika.BlockingConnection(pika.ConnectionParameters(
    address, 5672, f'{virtual_host}', credentials))
# ở đây là tạo ra kết nối đến RabbitMQ từ các "tham số kết nối"
channel = connection.channel()
# phải tạo ra channel từ kết nối được tạo ra, hình dung kết nối mới biết nhau thôi nhưng để truyền nhận thì cần tạo channel
# thì mới thực hiện các hoạt động publish (xuất) hoặc consume (tiêu thụ), hình dung channel như cái queue cái đường ống để làm

if __name__ == "__main__":
    src.Log.print_with_color(
        "[>>>] Client sending registration message to server...", "red")  # in ra màu đỏ

    # dữ liệu gửi đến server gồm action, client_id, layer_id và message
    # action: REGISTER, client_id: id của client, layer_id: id của layer, message: nội dung gửi đến server
    data = {"action": "REGISTER", "client_id": client_id,
            "layer_id": args.layer_id, "message": "Hello from Client!"}

    # đưa client, layer chia model, kênh truyền, thiết bị của client, mục đích là để thiết lập hàm để chạy inference, sau đó đưa hàm
    # inference này xuống cho client chạy
    # lập lịch ở đây hiểu là sẽ cho client này chạy trước hay chạy sau, rồi chọn
    scheduler = Scheduler(client_id, args.layer_id, channel, device)
    # hàm inference cho đúng, nó sẽ dựa vào layer_id

    # Rpclient liên quan trực tiếp đến hàng đợi rpc_queue
    client = RpcClient(client_id, args.layer_id, address, username,
                       password, virtual_host, scheduler.inference_func, device)
    # ở đây hiểu đơn giản là mới đưa thông số vào để thiết lập client dựa trên các yêu cầu đầu vào
    # thông tin là của 1 client thôi, vì chương trình này chỉ chạy 1 lần

    # client sẽ đẩy dữ liệu đến rpc_queue để server consuming liên tục, đây chính là yêu cầu mà client này muốn
    client.send_to_server(data)
    # REGISTER, và cứ có tin nhắn đến rpc_queue thì server sẽ thực hiện ngay on_request chính là yêu cầu

    # lắng nghe phản hồi từ server tức đợi server gửi đến cái hàng đợi reply_{id của client}, server gửi đến client nào
    client.wait_response()
    # thì sẽ gửi đến hàng đợi tương ứng đó, và client sẽ lắng nghe tin nhắn gửi đến hàng đợi của mình, tin nhắn đó chính là thông số của
    # model, thì client sẽ chạy và tính toán inference
