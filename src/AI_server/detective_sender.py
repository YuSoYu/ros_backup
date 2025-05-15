from ultralytics import YOLO
import socket, zlib, cv2, numpy as np
import struct  # 바이너리 전송용
import time


model = YOLO("/home/usou/dev_ws/final_backup/ros_backup/src/runs/detect/train8_person_roscar/weights/person_roscar_best.pt")

# 로봇 ID 수동 설정 (예: 1번)
ROSCAR_ID = 1

# UDP (영상 수신)
UDP_PORT = 9999
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.bind(('0.0.0.0', UDP_PORT))
print("YOLO 수신 서버 준비 완료")

# TCP (인식 결과 전송)
TCP_IP = "192.168.0.30"
TCP_PORT = 5001
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_sock.connect((TCP_IP, TCP_PORT))
print("Main Server 연결 완료")

# 클래스 → 결과 코드 매핑 정의
CLASS_CODE = {
    "person": 0x00,
    "roscar": 0x01
}

while True:
    data, _ = udp_sock.recvfrom(65536)
    try:
        decompressed = zlib.decompress(data)
        np_data = np.frombuffer(decompressed, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        results = model(frame, conf=0.4, verbose=False)
        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            if cls_name not in CLASS_CODE:
                continue  # 명세에 없는 클래스는 제외

            result_code = CLASS_CODE[cls_name]

            # 바이너리 메시지 생성: 2-byte 'IN' + 1-byte roscar_id + 1-byte result_code
            binary_msg = struct.pack('>2sBB', b'IN', ROSCAR_ID, result_code)

            try:
                tcp_sock.sendall(binary_msg)
                print(f"✅ 전송: IN | ID={ROSCAR_ID} | class={cls_name}({result_code})")
            except Exception as e:
                print("❌ TCP 전송 실패:", e)

        annotated = results[0].plot()
        cv2.imshow("YOLO Detection", annotated)
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("프레임 처리 오류:", e)
        continue

udp_sock.close()
tcp_sock.close()
cv2.destroyAllWindows()
