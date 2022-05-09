import socket
import cv2
import numpy as np
from queue import Queue
from _thread import *

"""
소켓 서버는 수신자 : 라즈베리파이 (GPU 서버측에서 발신이 올때 까지 queue에 frame 쌓임)
    --> stitching시 frame을 소실할 경우 특징점을 찾지 못하게 될 수도 있기에 
1. 소켓 생성 --> socket()
2. 포트 맵핑(바인딩) --> socket.bind(호스트이름, 포트번호) : Network layer 자원인 IP port
3. client의 발신 대기(리스닝) --> listen() : client(GPU 서버)로 부터 연결 요청 들어오면 return
    --> 해당 연결을 받아 들이기 위해 accept() 메소드를 호출
4. client 연결 받기 --> accept() : (socket,address info) 튜플 리넡
    --> 처음에 생성한 소켓과 별개의 객체로 클라이언트와 연결이 구성(실제 데이터를 주고 받을 수 있는 창구
    --> 해당 소켓은 연결이 들어와서 listen(), accept()가 호출될 때마다 생성가능
    --> 멀티 thread 처리기 1:다수 연결 처리 가능 + 대용량 data 처리 가능
    [참고논문] https://www.koreascience.or.kr/article/JAKO201314358630385.pdf

=========================================================================================
<동작 메커니즘>
1. socket 생성, 포트 binding, client(GPU server) 리스닝 , client accept() 
2. thread_camera 동작(frame읽어서 queue에 저장) : start_new_thread(thread_camera, (queue,)) # function이 thread 시작점 : start_new_thread(function, args) : 콜백함수 및 인자(튜플)
...
...
"""

queue = Queue()  # queue만들기


# 쓰레드 함수 :socket
def thread_socket(client_socket_info, address, queue):
    print('Connected by :', address[0], ':', address[1])

    while True:

        try:
            data = client_socket_info.recv(1024)  # 소켓으로 부터 데이터 읽기(버퍼크기byte)

            if not data:
                print('Disconnected by ' + address[0], ':', address[1])
                break

            stringData = queue.get()
            client_socket_info.send(str(len(stringData)).ljust(16).encode())
            client_socket_info.send(stringData)

        except ConnectionResetError as e:

            print('Disconnected by ' + address[0], ':', address[1])
            break

    client_socket_info.close()  # 소켓 닫기


# 쓰레드 함수 :camera
def thread_camera(queue):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret == False:
            continue

        fourcc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        # frame을 binary 형태로 변환 jpg로 decoding
        result, img_encode = cv2.imencode('.jpg', frame, fourcc)
        data = np.array(img_encode)  # numpy array로 안바꿔주면 ERROR
        stringData = data.tostring()  # byte 스트링으로 변경

        queue.put(stringData)  # queue에 byte stream 넣기
        # queue.put(img_encode)
        #cv2.imshow('image_server', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break


HOST = ''
PORT = 8090
# 소켓생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# setsockopt(소켓 레벨,이미 사용된 주소를 재사용(bind))
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 포트에 매핑(바인딩)
server_socket.bind((HOST, PORT))
# 리스닝(발신대기) : 클라이언트(GPU서버)가 bind된 port로 연결할 때 까지 기다리는 blocking 함수 , Blocking : 해당 프로세스가 bint된 port가 준비 될 때 까지 기다림
server_socket.listen()
print('server start')
start_new_thread(thread_camera, (queue,))  # function이 thread 시작점 : start_new_thread(function, args) : 콜백함수 및 인자(튜플)

while True:
    print('wait')
    client_socket_info, address = server_socket.accept()  # client socket에대한 (소켓정보, 주소정보)
    start_new_thread(thread_socket, (client_socket_info, address, queue,))
    # start_new_thread(function, args, kwargs)

cap.release()
cv2.destroyAllWindows()
server_socket.close()