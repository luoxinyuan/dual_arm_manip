import sys
root_dir = "../"
sys.path.append(root_dir)

from head import Head

import serial.tools.list_ports

# List all available serial ports
ports = list(serial.tools.list_ports.comports())
for port in ports:
    print(f"Port: {port.device}, Description: {port.description}")

head = Head.init_from_yaml(cfg_path=f'{root_dir}/cfg/cfg_head.yaml')

print(head)

head.servo_move(1000,1,400)
head.servo_move(1000,2,500)

head.disconnect()