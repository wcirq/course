# -*- coding: utf-8 -*- 
# @File conv2d.py
# @Time 2021/3/10 15:55
# @Author wcy
# @Software: PyCharm
# @Site
import cv2
import tensorflow as tf
import numpy as np
from scipy import signal


def test1():
    x = np.reshape(np.arange(1, 10), (3, 3))
    k = np.array([
        [-1, -2],
        [2, 1]
    ])
    y = signal.convolve2d(x, k, mode="full")
    print("----x----")
    print(x)
    print("----k----")
    print(k)
    print("----y----")
    print(y)


def test2():
    x = np.reshape(np.arange(1, 10), (3, 3))
    x = x.astype(dtype=np.float32)
    x = np.pad(x, (1, 1))
    # 以下卷积核未旋转180度
    # k = np.array([
    #     [-1, -2],
    #     [2, 1]
    # ])
    # 以下卷积核旋转180度
    k = np.array([
        [1, 2],
        [-2, -1]
    ], dtype=np.float32)
    y = tf.nn.conv2d(np.reshape(x, (1, 5, 5, 1)), np.reshape(k, (2, 2, 1, 1)), strides=[1, 1], padding="VALID")
    print("----x----")
    print(x)
    print("----k----")
    print(k)
    print("----y----")
    print(tf.squeeze(y).numpy())

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    k = tf.convert_to_tensor(k, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([k, x])
        y = tf.nn.conv2d(tf.reshape(x, (1, 5, 5, 1)), tf.reshape(k, (2, 2, 1, 1)), strides=[1, 1], padding="VALID")
    grads = tape.gradient(y, [k, x])
    print()


def test3():
    image = cv2.imread("images/cat.jpeg", 0)
    h, w = image.shape
    input = tf.cast(image, dtype=tf.float32)
    input = tf.reshape(input, (1, h, w, 1))
    # filters =np.array([
    #     [1, 2],
    #     [-2, -1]
    # ], dtype=np.float32)
    # filters = tf.reshape(filters, (2, 2, 1, 1))
    filters = np.ones((5, 5), dtype=np.float32)/25
    filters = tf.reshape(filters, (5, 5, 1, 1))
    output = tf.nn.conv2d(input, filters, strides=[1, 1], padding="VALID")
    output = tf.squeeze(output)
    output = output.numpy()
    output = (output-output.min())/(output.max()-output.min())
    cv2.imshow("image", image)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    print()


if __name__ == '__main__':
    # test1()
    # test2()
    test3()
