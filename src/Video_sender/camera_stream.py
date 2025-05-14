import socket
import cv2
import zlib
import time

# 전송 대상 서버 정보
SERVER_IP = "192.168.0.30"
SERVER_PORT = 9999

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 프레임 해상도 강제 설정 (안전한 전송을 위한 핵심)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60
MAX_UDP_SIZE = 60000  # 안전 한계치 설정

print("📷 camera_stream 송신 시작...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임 캡처 실패")
        continue

    # 해상도 강제 축소
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # JPEG 인코딩 + 압축
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success:
        print("⚠️ JPEG 인코딩 실패")
        continue

    compressed = zlib.compress(encoded.tobytes())

    # 전송 크기 확인 후 송신
    if len(compressed) < MAX_UDP_SIZE:
        try:
            sock.sendto(compressed, (SERVER_IP, SERVER_PORT))
            print(f"✅ 프레임 전송 완료 ({len(compressed)} bytes)")
        except Exception as e:
            print("❌ 송신 에러:", e)
    else:
        print(f"🚫 압축된 프레임 크기 초과: {len(compressed)} bytes (송신 생략)")

    # ESC 키 종료
    if cv2.waitKey(1) == 27:
        break

    # 💡 전송 속도 조절 (옵션)
    time.sleep(0.03)  # 약 30 FPS

# 자원 해제
cap.release()
sock.close()
print("📴 송신 종료.")
