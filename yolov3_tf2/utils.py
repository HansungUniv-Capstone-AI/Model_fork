from absl import logging
import numpy as np
import tensorflow as tf
import cv2
from apiRequestTester import send_api

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names, centerW, centerH):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    centerWidth , centerHeight = centerW, centerH

    wh = np.flip(img.shape[0:2])

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 2)
        className = '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i])
        img = cv2.putText(img, className,
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # class_names = class_names[int(classes[i])]
        centerWh = [centerWidth, centerHeight]
        x1y1Np = np.array(list(x1y1))
        x2y2Np = np.array(list(x2y2))
        centerWhNp = np.array(centerWh)
        ocWh = (x2y2Np + x1y1Np)/2 # object center width  height
        ocw,och = ocWh

        #print("center = " + str(centerWhNp))#str(centerWidth) + ", " + str(centerHeight))
        #print('class_name : ' + className + " /// left_top : " + str(x1y1) + " /// right_bottom : " + str(x2y2))
        #print("Object center = " + str(ocw) + ", "+ str(och))#str(ocWh))
        direction = centerWidth - ocw

        # #contours approximation
        # targetGray = cv2.cvtColor(x1y1, cv2.COLOR_BGR2GRAY)
        # shapesGray = cv2.cvtColor(x2y2, cv2.COLOR_BGR2GRAY)
        # # 바이너리 스케일 변환
        # ret, targetTh = cv2.threshold(targetGray, 127, 255, cv2.THRESH_BINARY_INV)
        # ret, shapesTh = cv2.threshold(shapesGray, 127, 255, cv2.THRESH_BINARY_INV)
        # # 컨투어 찾기
        # _, cntrs_target, _ = cv2.findContours(targetTh, cv2.RETR_EXTERNAL, \
        #                                       cv2.CHAIN_APPROX_SIMPLE)
        # _, cntrs_shapes, _ = cv2.findContours(shapesTh, cv2.RETR_EXTERNAL, \
        #                                       cv2.CHAIN_APPROX_SIMPLE)
        # #contour, 면적 계산  cv2.contourArea(contour 외곽선 좌표(numpy.ndarray.shape),부호있는 면적반환 )
        # target_area = cv2.contourArea(cntrs_target, True)
        # shapes_area = cv2.contourArea(cntrs_shapes, True)
        # distance = target_area - shapes_area

        if "person" in className : # 특정 클래스이고
            print("person class Object Detection")
            if direction > 100:  # 객체가 이미지의 센터보다 왼쪽에 위치
                print(str(direction) +" <<-- direction left ")
                #send_api("left", "POST")
            elif direction < -100: # 객체가 이미지의 센터보다 오른쪽에 위치
                print(str(direction) +" direction right -->> ")
                #send_api("right", "POST")

            # 면적으로 앞뒤 조정 ?di
            # elif distance > 0: # 객체가 기존면적보다 작다 = 멀다
            #     print(str(distance) +" distance go ahead")
            #     send_api("go", "POST")
            # elif distance < 0:  # 객체가 기존면적보다 크다 = 가깝다
            #     print(str(distance) + " distance stop")
            #     send_api("stop", "POST")
            #
            else:
                print("direction straight")
                send_api("straight", "POST")


    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
