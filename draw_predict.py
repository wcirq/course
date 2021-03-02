# -*- coding: utf-8 -*-
# @File draw_predict.py
# @Time 2021/3/1 下午3:36
# @Author wcirq
# @Software PyCharm
# @Site
import cv2
import numpy as np
import tensorflow as tf

model_path = "resources/model/model.h5"


class Model(object):

    def __init__(self):
        self.model = tf.keras.models.load_model("resources/model/model.h5")

    def predict(self, image):
        image = cv2.resize(image, (28, 28))
        image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        inputs = tf.reshape(image, (-1, 28 * 28))
        outputs = self.model(inputs)
        index = tf.argmax(outputs, axis=1).numpy()[0]
        score = tf.reduce_max(outputs, axis=1).numpy()[0]
        return index, score


class Draw(object):

    def __init__(self):
        self.drawing = False
        self.model = Model()
        self.frame = np.zeros((200, 200, 3), np.uint8)
        self.window_name = 'image'
        cv2.namedWindow(self.window_name)
        # create switch for ON/OFF functionality
        self.switch1 = '0 : pen \n1 : clear'
        self.switch2 = '0: draw \n1: predict '
        cv2.createTrackbar(self.switch1, self.window_name, 0, 1, self.nothing)
        cv2.createTrackbar(self.switch2, self.window_name, 0, 1, self.identify)

    def nothing(self, x):
        pass

    def identify(self, flag):
        if flag == 1:
            image = self.frame[..., 0]
            index, score = self.model.predict(image)
            print("预测结果 ", index, "score", score)

    def draw_circle(self, event, x, y, flags, param):
        """
        鼠标回调方法，绘制线条
        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        """
        global ix, iy, drawing
        g = 255
        b = 255
        r = 255
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.frame, (x, y), 10, (g, b, r), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.circle(self.frame, (x, y), 10, (g, b, r), -1)

    def loop(self):
        while 1:
            cv2.imshow('image', self.frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            s = cv2.getTrackbarPos(self.switch1, self.window_name)
            if s == 1:
                self.frame[:] = 0
            else:
                cv2.setMouseCallback(self.window_name, self.draw_circle)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    draw = Draw()
    draw.loop()
