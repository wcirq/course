# -*- coding: utf-8 -*-
# @File main.py
# @Time 2021/3/18 下午9:46
# @Author wcirq
# @Software PyCharm
# @Site
import cv2
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    image = tf.convert_to_tensor(np.array([
        [1, 0],
        [0, 0]
    ], dtype=np.float32).reshape((2, 2, 1)))
    # image = tf.zeros((3, 3, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([image])
        new_image1 = tf.image.resize(image, (4, 4), method=tf.image.ResizeMethod.BICUBIC)
        new_image2 = tf.image.resize(image, (4, 4), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        new_image3 = tf.image.resize(image, (4, 4), method=tf.image.ResizeMethod.BILINEAR)

    # [grads] = tape.gradient(new_image1, [image])
    [grads] = tape.gradient(new_image2, [image])
    # [grads] = tape.gradient(new_image3, [image])
    grads_np = grads[..., 0].numpy()
    img = image[..., 0].numpy()
    img1 = new_image1[..., 0].numpy()
    img2 = new_image2[..., 0].numpy()
    img3 = new_image3[..., 0].numpy()
    print()
