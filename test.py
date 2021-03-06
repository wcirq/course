# -*- coding: utf-8 -*-
# @File test.py
# @Time 2021/3/1 下午2:21
# @Author wcirq
# @Software PyCharm
# @Site
import cv2
import tensorflow as tf
from tensorflow.keras import datasets


def show_image(image):
    win = "image"
    image = cv2.resize(image, (200, 200))
    cv2.imshow(win, image)
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    cv2.destroyWindow(win)


def main():
    model_path = "resources/model/model_conv2d.h5"
    model = tf.keras.models.load_model(model_path)
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    for i in range(len(x_val)):
        inputs = x_val[i:i+1]
        image = inputs[0]
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) / 255.
        if "conv2d" in model_path:
            inputs = tf.reshape(inputs, (-1, 28, 28, 1))
        else:
            inputs = tf.reshape(inputs, (-1, 28 * 28))
        outputs = model(inputs)
        index = tf.argmax(outputs, axis=1).numpy()[0]
        print("模型预测为", index)
        show_image(image)


if __name__ == '__main__':
    main()
