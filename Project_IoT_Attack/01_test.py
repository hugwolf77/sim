## https://www.wireshark.org/download.html
## https://npcap.com/#download

import os
import pandas as pd
import pyshark as sk

def print_packet_tree(packet, indent=0):
    """패킷의 계층 구조를 트리 형태로 출력하는 함수"""
    print("  " * indent + f"Packet {packet.number} (Length: {packet.length}, Time: {packet.sniff_time})")
    for layer in packet.layers:
        print("  " * (indent + 1) + f"Layer: {layer.layer_name}")
        for field in layer.field_names:
            try:
                field_value = layer.get_field_value(field)
                print("  " * (indent + 2) + f"{field}: {field_value}")
            except AttributeError:
                pass

train_benign = '/home/augustine77/mylab/sim/sim/Pyshark/data/CIC_2025/bluetooth/attacks/pcap/train/Bluetooth_Benign_train.pcap'
profiling = '/home/augustine77/mylab/sim/sim/Pyshark/data/CIC_2025/bluetooth/profiling/pcap/Checkme_O2_Oximeter_Power.pcap'
cap = sk.FileCapture(input_file=profiling)


# 첫 번째 패킷 가져오기 (인덱싱 사용)
first_packet = cap[0]


# 패킷 트리 출력
print_packet_tree(first_packet)

# FileCapture 객체 닫기
cap.close()


# # 패킷 정보 추출
# packet_data = {}

# # 기본 정보
# packet_data['number'] = first_packet.number
# packet_data['length'] = first_packet.length
# packet_data['time'] = first_packet.sniff_time

# # 레이어 정보 추출
# for layer in first_packet.layers:
#     layer_name = layer.layer_name
#     for field in layer.field_names:
#         try:
#             packet_data[f'{layer_name}.{field}'] = layer.get_field_value(field)
#         except AttributeError:
#             pass

# # 데이터프레임 생성
# df = pd.DataFrame([packet_data])

# # 데이터프레임 출력
# print(df)





# # 모든 패킷 순회
# for packet in cap:
#     # 패킷 정보 출력
#     print(packet)
#     raise

#     # 특정 프로토콜 정보 접근 (예: IP)
#     if 'IP' in packet:
#         print(f"Source IP: {packet.ip.src}")
#         print(f"Destination IP: {packet.ip.dst}")

#     # 특정 레이어 정보 접근 (예: TCP)
#     if 'TCP' in packet:
#         print(f"Source Port: {packet.tcp.srcport}")
#         print(f"Destination Port: {packet.tcp.dstport}")

# # 파일 닫기
# cap.close()