import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import socket
import numpy as np


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')



# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b'' #바이트(인코딩 지정) 객체 생성
    while count: #지정한 바이트 길이까지만 받기
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

########################
HOST = ''
PORT = 8888 # client <-> server간 포트 동일하게
# UDP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # SOCK_STREAM : TCP
print('Socket created')
# 서버의 아이피와 포트번호 지정 -> 포트에 매핑(바인딩)
s.bind((HOST, PORT))
print('Socket bind complete')
# 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다) : 클라이언트가 bind된 port로 연결할 때까지 기다리는 blocking 함수
s.listen(10)
print('Socket now listening')
# 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
conn, addr = s.accept()
#########################

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(0)#int(FLAGS.video)) # vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video) # vid = cv2.VideoCapture(FLAGS.video)

    out = None

    #if FLAGS.output:
        # by default VideoCapture returns float instead of int
     #   width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
      #  height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
       # fps = int(vid.get(cv2.CAP_PROP_FPS))
       # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
       # out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    centerWidth = int(width/2)
    centerHeight = int(height/2)

    while True:
        # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype='uint8')

        # data를 디코딩한다.
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]

        frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names,centerWidth,centerHeight)
        frame = cv2.putText(frame, "Time: {:.2f}ms".format(sum(times) / len(times) * 1000), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if FLAGS.output:
            out.write(frame)

        cv2.imshow('Realtimetcp', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    """
    while True:
        _, frame = vid.read()

        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
        frame = cv2.putText(frame, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(frame)
        cv2.imshow('output', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        """
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass