# -*- coding: utf-8 -*-
# @File fangsebianhuan.py
# @Time 2021/3/17 上午12:06
# @Author wcirq
# @Software PyCharm
# @Site
import cv2
import numpy as np


def rotate(image, angle):
    """
    旋转图片
    :param image:
    :param angle:
    :return:
    """
    PI = np.pi
    heightNew = int(image.shape[1] * np.abs(np.sin(angle * PI / 180)) + image.shape[0] * np.abs(np.cos(angle * PI / 180)))
    widthNew = int(image.shape[0] * np.abs(np.sin(angle * PI / 180)) + image.shape[1] * np.abs(np.cos(angle * PI / 180)))
    # pt = (image.shape[1] / 2., image.shape[0] / 2.)
    pt = (2., 1.)
    # 旋转矩阵
    # --                      --
    # | ∂, β, (1 -∂) * pt.x - β * pt.y |
    # | -β, ∂, β * pt.x + (1 -∂) * pt.y |
    # --                      --
    # 其中 ∂=scale * cos(angle), β = scale * sin(angle)

    ### getRotationMatrix2D 的实现 ###
    scale = 1
    a = scale * np.cos(angle * PI / 180)
    b = scale * np.sin(angle * PI / 180)
    r1 = np.array([[a, b, (1 - a) * pt[0] - b * pt[1]],
                   [-b, a, b * pt[0] + (1 - a) * pt[1]]])
    ### getRotationMatrix2D 的实现 ###

    r = cv2.getRotationMatrix2D(pt, angle, scale)  # 获取旋转矩阵(旋转中心(pt), 旋转角度(angle)， 缩放系数(scale)
    r[0, 2] += (widthNew - image.shape[1]) / 2
    r[1, 2] += (heightNew - image.shape[0]) / 2
    dst = cv2.warpAffine(image, r, (widthNew, heightNew), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)  # 进行仿射变换（输入图像, 2X3的变换矩阵, 指定图像输出尺寸, 插值算法标识符, 边界填充BORDER_REPLICATE)
    return dst


if __name__ == '__main__':
    # image = cv2.imread("data/images/img_1001.jpg")
    # dst = rotate(image, 45)
    # cv2.imshow("image", image)
    # cv2.imshow("dst", dst)
    # cv2.waitKey(0)

    # 点（4,1）绕点（2,1）逆时针旋转45度
    xy0 = np.array([4, 1, 1]).reshape((3, 1))
    angle = 45
    # 绕该点旋转（center_x， center_y）
    center_x = 2
    center_y = 1

    matrix_translation1 = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    radian = angle * np.pi / 180
    matrix_rotate = np.array([
        [np.cos(radian), -np.sin(radian), 0],
        [np.sin(radian), np.cos(radian), 0],
        [0, 0, 1]
    ])
    matrix_translation2 = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    matrix = np.dot(matrix_translation1, matrix_rotate)
    matrix = np.dot(matrix, matrix_translation2)

    scale = 1
    a = scale * np.cos(angle * np.pi / 180)
    b = scale * np.sin(angle * np.pi / 180)
    r1 = np.array([[a, b, (1 - a) * xy0[0][0] - b * xy0[1][0]],
                   [-b, a, b * xy0[0][0] + (1 - a) * xy0[1][0]]])

    xy1 = np.dot(matrix, xy0)
    print(matrix)
    print(xy0)
    print(xy1)
