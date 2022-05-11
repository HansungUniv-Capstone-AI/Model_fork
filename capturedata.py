import cv2
import time

cap = cv2.VideoCapture('./video1.mp4')
FPS = 15
prev_time = 0
count = 0

while True:
    ret, image = cap.read()

    if not ret:
        break

    current_time = time.time() - prev_time

    if (ret is True) and (current_time > 1. / FPS):
        prev_time = time.time()

        cv2.imwrite('./image/' + str(count) + '.jpg', image)
        count += 1