# -*- coding: utf-8 -*-
# @File fangsebianhuan.py
# @Time 2021/3/17 上午12:06
# @Author wcirq
# @Software PyCharm
# @Site
import itertools
import time
import tensorflow as tf
import cv2
import numpy as np
import pylab as pl
from matplotlib import collections as mc, pyplot as plt


def my_warp_affine_nearest_neighbor(image, matrix, border_constant=True, constant=127):
    """
    实现最近邻插值算法
    :param image: 待变换图片
    :param matrix: 仿射矩阵
    :param border_constant: True：边缘用0填充， False: 边缘用最近的值填充
    :return:
    """
    h, w, c = image.shape
    # 求matrix得逆矩阵
    matrix_inv = tf.linalg.inv(matrix)
    # 生成变换后图片的所有点的索引 shape [3, h*w]
    x, y = tf.meshgrid(np.arange(0, w).astype(np.float32), np.arange(0, h).astype(np.float32))
    x = tf.expand_dims(x, axis=2)
    y = tf.expand_dims(y, axis=2)
    homogeneous = tf.ones((h, w, 1), dtype=x.dtype)
    xy1 = tf.concat((x, y, homogeneous), axis=2)
    xy1 = tf.transpose(tf.reshape(xy1, (-1, 3)))
    # 计算变换后的图片像素索引对应的原来图片像素索引
    xy0 = tf.transpose(tf.matmul(matrix_inv, xy1))
    start = time.time()
    # -------- 利用tensorflow实现实现最近邻(会快很多) --------
    # 利用四舍五入实现最近邻
    index = tf.cast(tf.round(tf.transpose(xy0)), dtype=tf.int32)[:2, :]
    # 将超出原始图片坐标范围的点全部用最近的点表示
    index_0 = tf.clip_by_value(index[0], clip_value_min=0, clip_value_max=w - 1)
    index_1 = tf.clip_by_value(index[1], clip_value_min=0, clip_value_max=h - 1)
    indices = tf.concat((index_1[:, tf.newaxis], index_0[:, tf.newaxis]), axis=1)
    empty_image = tf.gather_nd(image, indices)
    if border_constant:
        bool_beyond = ((index < 0) | (index[0, :] > w - 1) | (index[1, :] > h - 1))
        condition = tf.logical_or(bool_beyond[0, :], bool_beyond[1, :])
        beyond_image = tf.ones_like(empty_image, dtype=tf.float32)*constant
        empty_image = tf.where(condition[:, tf.newaxis], x=beyond_image, y=empty_image)
    empty_image = tf.reshape(empty_image, (h, w, c))
    # -------- 利用tensorflow实现实现最近邻 --------
    # print(time.time() - start)
    return empty_image


def my_warp_affine_bilinear(image, matrix, border_constant=True, constant=0):
    """
    实现双线性插值算法
    :param image: 待变换图片
    :param matrix: 仿射矩阵
    :param border_constant: True：边缘用0填充， False: 边缘用最近的值填充
    :return:
    """
    h, w, c = image.shape
    # 求matrix得逆矩阵
    matrix_inv = tf.linalg.inv(matrix)
    # 生成变换后图片的所有点的索引 shape [3, h*w]
    x, y = tf.meshgrid(np.arange(0, w).astype(np.float32), np.arange(0, h).astype(np.float32))
    x = tf.expand_dims(x, axis=2)
    y = tf.expand_dims(y, axis=2)
    homogeneous = tf.ones((h, w, 1), dtype=x.dtype)
    xy1 = tf.concat((x, y, homogeneous), axis=2)
    xy1 = tf.transpose(tf.reshape(xy1, (-1, 3)))
    # 计算变换后的图片像素索引对应的原来图片像素索引
    xy0 = tf.transpose(tf.matmul(matrix_inv, xy1))
    start = time.time()

    # --------- tensorflow 实现双线性插值 ------
    xy0 = xy0[:, :2]
    min_xy = tf.cast(tf.floor(xy0), dtype=tf.int32)[:, :2]
    max_xy = min_xy + 1  # 不能用 np.ceil(xy0), 因为为整数时不会向上取整

    min_xy = tf.cast(min_xy, tf.float32)
    max_xy = tf.cast(max_xy, tf.float32)

    min_xy_0 = tf.clip_by_value(min_xy[:, 0], clip_value_min=0, clip_value_max=w - 1)
    min_xy_1 = tf.clip_by_value(min_xy[:, 1], clip_value_min=0, clip_value_max=h - 1)
    max_xy_0 = tf.clip_by_value(max_xy[:, 0], clip_value_min=0, clip_value_max=w - 1)
    max_xy_1 = tf.clip_by_value(max_xy[:, 1], clip_value_min=0, clip_value_max=h - 1)

    min_xy_0 = tf.cast(min_xy_0, tf.int32)
    min_xy_1 = tf.cast(min_xy_1, tf.int32)
    max_xy_0 = tf.cast(max_xy_0, tf.int32)
    max_xy_1 = tf.cast(max_xy_1, tf.int32)

    w0_xy = (xy0 - min_xy) / (max_xy - min_xy)
    w1_xy = (max_xy - xy0) / (max_xy - min_xy)
    p0 = tf.gather_nd(image, tf.concat((min_xy_1[..., tf.newaxis], min_xy_0[..., tf.newaxis]), axis=1))
    p1 = tf.gather_nd(image, tf.concat((min_xy_1[..., tf.newaxis], max_xy_0[..., tf.newaxis]), axis=1))
    p2 = tf.gather_nd(image, tf.concat((max_xy_1[..., tf.newaxis], max_xy_0[..., tf.newaxis]), axis=1))
    p3 = tf.gather_nd(image, tf.concat((max_xy_1[..., tf.newaxis], min_xy_0[..., tf.newaxis]), axis=1))
    r0 = w0_xy[:, :1] * p1 + w1_xy[:, :1] * p0
    r1 = w0_xy[:, :1] * p2 + w1_xy[:, :1] * p3
    empty_image = w0_xy[:, 1:] * r1 + w1_xy[:, 1:] * r0
    if border_constant:
        bool_beyond = ((xy0 < 0) | (xy0[:, :1] > w - 1) | (xy0[:, 1:] > h - 1))
        condition = tf.logical_or(bool_beyond[:, 0], bool_beyond[:, 1])
        beyond_image = tf.ones_like(empty_image, dtype=tf.float32)*constant
        empty_image = tf.where(condition[:, tf.newaxis], x=beyond_image, y=empty_image)
    empty_image = tf.reshape(empty_image, (h, w, c))
    empty_image = tf.cast(empty_image, dtype=tf.uint8)
    # --------- tensorflow 实现双线性插值 ------
    # print(time.time() - start)
    return empty_image


def rotate(image, angle, scale=0.5, algorithm=0, border_constant=True, constant=127):
    """
    图片绕图片中心点旋转指定角度
    :param image: 待旋转的图片
    :param angle: 旋转角度 （顺时针方向为正）
    :param algorithm: 使用的填充算法 0-最近邻 1-双线性插值
    :return:
    """
    center_x, center_y = image.shape[1] / 2., image.shape[0] / 2.

    matrix_translation1 = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    matrix_translation1 = tf.convert_to_tensor(matrix_translation1, dtype=tf.float32)
    radian = angle * np.pi / 180
    matrix_rotate = np.array([
        [np.cos(radian), np.sin(radian), 0],
        [-np.sin(radian), np.cos(radian), 0],
        [0, 0, 1]
    ])
    matrix_rotate = tf.convert_to_tensor(matrix_rotate, dtype=tf.float32)
    matrix_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    matrix_scale = tf.convert_to_tensor(matrix_scale, dtype=tf.float32)
    matrix_translation2 = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    matrix_translation2 = tf.convert_to_tensor(matrix_translation2, dtype=tf.float32)

    matrix = np.diag(np.ones((3,)))
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)

    matrix = tf.matmul(matrix, matrix_translation1)
    matrix = tf.matmul(matrix, matrix_rotate)
    matrix = tf.matmul(matrix, matrix_scale)
    matrix = tf.matmul(matrix, matrix_translation2)

    if algorithm == 0:
        dst = my_warp_affine_nearest_neighbor(image, matrix, border_constant=border_constant, constant=constant)
    else:
        dst = my_warp_affine_bilinear(image, matrix, border_constant=border_constant, constant=constant)
    return dst


def resize_image_example():
    # image = cv2.imread("data/images/img_1001.jpg")
    image = cv2.imread("/home/wcirq/Pictures/meinv.jpg")
    image = np.transpose(image, (1, 0, 2))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    dst1 = rotate(image, 45, scale=1.0, algorithm=0, border_constant=True, constant=255)
    dst2 = rotate(image, 45, scale=1.0, algorithm=1, border_constant=True, constant=255)
    cv2.imshow("image", image.numpy().astype(np.uint8))
    cv2.imshow("dst1", dst1.numpy().astype(np.uint8))
    cv2.imshow("dst2", dst2.numpy().astype(np.uint8))
    cv2.waitKey(0)


def vedio():
    cap = cv2.VideoCapture(0)
    angle = 0.0
    scale = 1.0
    while True:
        start = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        scale = float(np.sin(angle*np.pi/180)+1.000001)
        image = tf.convert_to_tensor(frame, dtype=tf.float32)
        image = rotate(image, angle, scale=scale, algorithm=0, border_constant=True, constant=255)
        # image = rotate(image, angle, scale=scale, algorithm=1, border_constant=True, constant=255)
        angle+=1
        frame = np.hstack((frame, image.numpy().astype(np.uint8)))
        end =time.time()
        cv2.putText(frame, f"FPS:{1//(end-start)}", (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    # resize_image_example()
    vedio()
