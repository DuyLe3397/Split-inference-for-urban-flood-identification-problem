import pika
from requests.auth import HTTPBasicAuth
import requests


# mục đích là dọn dẹp các hàng đợi cũ trong RabbitMQ
def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues'
    # kết nối đến RabbitMQ để lấy danh sách các hàng đợi
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:  # duyệt qua từng hàng đợi trong danh sách hàng đợi
            queue_name = queue['name']  # lấy tên hàng đợi
            # kiểm tra tên hàng đợi có nằm trong danh sách các hàng đợi cần xóa hay không
            # xóa tên hàng đợi bắt đầu bằng 1 chuỗi con như dưới không
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "prediction_queue") or queue_name.startswith("rpc_queue"):

                # xóa vĩnh viễn hàng đợi cũ
                http_channel.queue_delete(queue=queue_name)

            else:
                # xóa các message nhưng vẫn giữ hàng đợi
                http_channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    else:
        return False
