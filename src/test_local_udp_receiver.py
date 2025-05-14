import socket
import zlib
import cv2
import numpy as np

PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', PORT))

print("로컬 영상 수신 대기 중...")

while True:
    data, _ = sock.recvfrom(65536)  # 64KB 제한
    try:
        decompressed = zlib.decompress(data)
        np_data = np.frombuffer(decompressed, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow('Live from Pi Camera', frame)

        if cv2.waitKey(1) == 27:  # ESC 종료
            break
    except Exception as e:
        print("프레임 처리 에러:", e)
        continue

sock.close()
cv2.destroyAllWindows()
