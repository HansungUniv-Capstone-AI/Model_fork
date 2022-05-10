# -*- coding: utf8 -*-
import cv2
import socket
import numpy as np
import json
import requests
import threading
import time
import logging
import tensorflow as tf

from yolov3_tf2.dataset import transform_images
test = True
# 서버 url
getUrl = 'https://lbvvzx89g3.execute-api.ap-northeast-2.amazonaws.com/prod'
patchUrl = 'https://lbvvzx89g3.execute-api.ap-northeast-2.amazonaws.com/prod'


# 스레드에서 실행하는 함수
def statusUsedCheck():
    logging.info("[Thread] : start thread.")
    time.sleep(3)  # 3초
    # GET
    r = requests.get(getUrl)
    rJson = r.json()
    cartStatus = rJson['response']['cartStatus']
    used = rJson['response']['used']
    print("before patch used :  " + str(used))
    logging.info("[Thread] : exit thread.")
    return used

# 카트 사용 상태를 변경 used = True
def patchCartUsedStatus():
    headers = {'Content-type': 'application/json; charset=utf-8'}
    data = {
        'cartUsed': True,
    }
    r2 = requests.patch(patchUrl, headers=headers, data=json.dumps(data))
    # r2 = requests.get(patchUrl)
    rJson2 = r2.json()
    print("json result : " + str(rJson2))
    used = rJson2['response']['used']
    print("after patch used :  " + str(used))

#startStatus = statusUsedCheck()

## TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
## server ip, port
s.connect(('localhost', 9999))

## webcam 이미지 capture
cam = cv2.VideoCapture(0)

## 이미지 속성 변경 3 = width, 4 = height
cam.set(3, 640); # 640 320
cam.set(4, 480); # 480 240

## 0~100에서 90의 이미지 품질로 설정 (default = 95)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

if __name__ == "__main__":
    if statusUsedCheck() == False:  # 해당 카트를 사용중이지 않을때만
        # 사용을 시작하면 상태값 True로 변경하여 다른 사용자가 사용 x
        patchCartUsedStatus()

        while True:
            # 비디오의 한 프레임씩 읽는다.
            # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
            ret, img = cam.read()

            if img is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                continue

            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, 416)

            # cv2. imencode(ext, img [, params])
            # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
            result, img = cv2.imencode('.jpg', img, encode_param)
            # frame을 String 형태로 변환
            data = np.array(img)
            stringData = data.tostring()

            # 서버에 데이터 전송
            # (str(len(stringData))).encode().ljust(16)
            s.sendall((str(len(stringData))).encode().ljust(16) + stringData)
    else:
        logging.info("해당 카트는 사용중입니다.")
        print("해당 카트는 사용중입니다.")

    cam.release()