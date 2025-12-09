import argparse
import sys
import signal
from src.Server import Server
from src.Utils import delete_old_queues
import src.Log
import yaml

parser = argparse.ArgumentParser(
    description="Split learning framework with controller.")

parser.add_argument('--val', type=bool, required=False,
                    help='Mode validation model')

parser.add_argument('--test', type=bool, required=False,
                    help='Mode test model')

parser.add_argument('--compare', type=bool, required=False,
                    help='Mode compare model')

args = parser.parse_args()

file = open('config.yaml', encoding="utf8")
config = yaml.safe_load(file)
# with open('config.yaml') as file: # mở file cấu hình tự định nghĩa
#     config = yaml.safe_load(file)

address = config["rabbit"]["address"]  # địa chỉ RabbitMQ
username = config["rabbit"]["username"]  # tên đăng nhập RabbitMQ
password = config["rabbit"]["password"]  # mật khẩu RabbitMQ
virtual_host = config["rabbit"]["virtual-host"]  # virtual host RabbitMQ


def signal_handler():
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(address, username, password, virtual_host)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    delete_old_queues(address, username, password, virtual_host)
    # đưa cấu hình đã định nghĩa vào server để tạo ra 1 server mới
    server = Server(config, args.val, args.test, args.compare)
    server.start()  # khởi động server
    src.Log.print_with_color("Ok, ready!", "green")  # in ra màu xanh
