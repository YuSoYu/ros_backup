from ultralytics import YOLO
import socket, zlib, cv2, numpy as np
import json, time

model = YOLO("/home/usou/dev_ws/final_yolo_test/src/runs/detect/train8(person+roscar)/weights/person+roscar_best.pt")

UDP_PORT = 9999
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.bind(('0.0.0.0', UDP_PORT))
print("YOLO 수신 서버....")

TCP_IP = "192.168.0.30"
TCP_PORT = 5001
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_sock.connect((TCP_IP, TCP_PORT))
print("TCP 메시지 전송 채널 연결 완료")

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

        objects = []
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            objects.append({
                "id": f"{cls_name}_{i}",
                "class": cls_name,
                "conf": round(conf, 2),
                "bbox": [x1, y1, x2, y2],
                "timestamp": int(time.time())
            })

        if objects:
            msg = json.dumps(objects) + "\n"
            tcp_sock.sendall(msg.encode())

        annotated = results[0].plot()
        cv2.imshow("YOLO Detection", annotated)
        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("처리 에러:", e)
        continue

udp_sock.close()
tcp_sock.close()
cv2.destroyAllWindows()
