import socket
import json

HOST = "0.0.0.0"
PORT = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("메시지 수신 서버 대기중 ...")
conn, addr = sock.accept()
print(f"{addr} 연결됨")

buffer = b""

while True:
    data = conn.recv(1024)
    if not data:
        break

    buffer += data
    while b'\n' in buffer:
        line, buffer = buffer.split(b'\n', 1)
        try:
            msg = json.loads(line.decode())
            print("▶ 객체 수신: ", msg)

        except json.JSONDecodeError:
            print("❌ 메시지 파싱 오류 : ", line)

conn.close()
sock.close()
