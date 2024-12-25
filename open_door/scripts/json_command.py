import socket
import json

# 机械臂的 IP 和端口
ip = "192.168.10.18"  # 替换为机械臂的实际 IP
port = 8080          # 替换为机械臂的实际端口

# JSON 指令
command = {
    "command": "get_controller_state"  # 替换为你的实际指令
}
command_str = json.dumps(command) + "\r\n"  # 以 \r\n 结尾

try:
    # 创建 TCP 套接字
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 设置超时时间为 5 秒
        s.settimeout(5)

        # 连接到机械臂
        s.connect((ip, port))
        print(f"Connected to {ip}:{port}")

        # 发送 JSON 指令
        s.sendall(command_str.encode('utf-8'))
        print(f"Sent: {command_str.strip()}")

        # 尝试接收机械臂的响应
        try:
            response = s.recv(1024)  # 设置接收缓冲区大小
            print("Received:", response.decode('utf-8'))
        except socket.timeout:
            print("No response received within the timeout period.")

except socket.timeout:
    print("Connection attempt timed out.")
except socket.error as e:
    print(f"Socket error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    print("Program finished.")
