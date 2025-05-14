from ultralytics import YOLO
import socket, zlib, cv2, numpy as np

model = YOLO("/home/usou/dev_ws/final_yolo_test/src/runs/detect/train8(person+roscar)/weights/best.pt")  # 학습된 모델 적용

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

        results = model(frame, conf = 0.8,  verbose=False)
        annotated = results[0].plot()
        cv2.imshow('YOLO Detection', annotated)

        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("처리 에러:", e)
        continue

sock.close()
cv2.destroyAllWindows()
