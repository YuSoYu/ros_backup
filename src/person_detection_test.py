import socket
import zlib
import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLOv8 모델 로딩 (직접 학습한 모델 경로 사용)
model = YOLO("/home/usou/dev_ws/final_yolo_test/src/runs/detect/train6/weights/best.pt")

# 2. UDP 수신 소켓
PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', PORT))

print("YOLO 수신 서버 시작...")

while True:
    data, _ = sock.recvfrom(65536)
    try:
        decompressed = zlib.decompress(data)
        np_data = np.frombuffer(decompressed, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # 3. YOLO 추론
        results = model(frame, verbose=False)
        annotated = results[0].plot()  # bounding box 그려진 이미지 반환

        # 4. 화면에 출력
        cv2.imshow('YOLO Detection', annotated)

        if cv2.waitKey(1) == 27:  # ESC 키로 종료
            break

    except Exception as e:
        print("처리 에러:", e)
        continue

sock.close()
cv2.destroyAllWindows()