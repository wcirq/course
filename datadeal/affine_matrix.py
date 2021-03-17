# -*- coding: utf-8 -*-
# @File fangsebianhuan.py
# @Time 2021/3/17 上午12:06
# @Author wcirq
# @Software PyCharm
# @Site
import cv2
import numpy as np
import pylab as pl
from matplotlib import collections as mc, pyplot as plt


def rotate(image, angle):
    """
    图片绕图片中心点旋转指定角度
    :param image: 待旋转的图片
    :param angle: 旋转角度 （顺时针方向为正）
    :return:
    """
    PI = np.pi
    heightNew = int(
        image.shape[1] * np.abs(np.sin(angle * PI / 180)) + image.shape[0] * np.abs(np.cos(angle * PI / 180)))
    widthNew = int(
        image.shape[0] * np.abs(np.sin(angle * PI / 180)) + image.shape[1] * np.abs(np.cos(angle * PI / 180)))
    pt = (image.shape[1] / 2., image.shape[0] / 2.)
    # 绕任一点旋转矩阵
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
    # r[0, 2] += (widthNew - image.shape[1]) / 2
    # r[1, 2] += (heightNew - image.shape[0]) / 2
    dst = cv2.warpAffine(image, r, (widthNew, heightNew), cv2.INTER_LINEAR,
                         cv2.BORDER_CONSTANT)  # 进行仿射变换（输入图像, 2X3的变换矩阵, 指定图像输出尺寸, 插值算法标识符, 边界填充BORDER_REPLICATE)
    return dst


def affine_matrix_example():
    """二维坐标的仿射变换例子"""
    # 点（4,1）绕点（center_x,center_y）逆时针旋转 angle 度， 然后缩放 scale 倍。
    # 可以分解为：
    # 1.将（center_x,center_y）点平移到原点
    # 2.点（4,1）绕点原点逆时针旋转angle度
    # 3.将（center_x,center_y）点平移到回原来的位置
    # 4.再将得到的坐标乘以缩放比例scale
    xy0 = np.array([4, 1, 1]).reshape((3, 1))
    angle = 90
    center_x = 2
    center_y = 1
    scale = 2

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
    matrix_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    matrix_translation2 = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    matrix = np.diag(np.ones((3,)))  # shape (3, 3), 左上角到右下角对角线为1，其余为0的3X3矩阵

    matrix = np.dot(matrix, matrix_scale)  # 4
    matrix = np.dot(matrix, matrix_translation1)  # 3
    matrix = np.dot(matrix, matrix_rotate)  # 2
    matrix = np.dot(matrix, matrix_translation2)  # 1

    xy1 = np.dot(matrix, xy0)

    print(matrix)
    print(xy0)
    print(xy1)
    # 定义多条线条
    lines = [[(center_x, center_y), (xy0[0, 0], xy0[1, 0])],
             [(center_x, center_y), (xy1[0, 0], xy1[1, 0])]]
    # 定义线条颜色
    colors = np.array([(1, 0, 0, 1), (0, 1, 0, 1)])
    lc = mc.LineCollection(lines, colors=colors, linewidths=2, linestyles="solid")
    # 开始绘制线条
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    # 显示三个点的坐标
    plt.text(center_x, center_y, (center_x, center_y), color='b')
    plt.text(xy0[0, 0], xy0[1, 0], (xy0[0, 0], xy0[1, 0]), color='r')
    plt.text(xy1[0, 0], xy1[1, 0], (xy1[0, 0], xy1[1, 0]), color='g')
    # 设置横纵坐标轴的最小值和最大值
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.show()


def resize_image_example():
    image = cv2.imread("data/images/img_1001.jpg")
    dst = rotate(image, 45)
    cv2.imshow("image", image)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    affine_matrix_example()
    resize_image_example()

