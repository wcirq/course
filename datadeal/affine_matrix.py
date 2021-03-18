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


def rotate(image, angle, scale=0.5):
    """
    图片绕图片中心点旋转指定角度
    :param image: 待旋转的图片
    :param angle: 旋转角度 （顺时针方向为正）
    :return:
    """
    center_x, center_y = image.shape[1] / 2., image.shape[0] / 2.

    matrix_translation1 = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    radian = angle * np.pi / 180
    matrix_rotate = np.array([
        [np.cos(radian), np.sin(radian), 0],
        [-np.sin(radian), np.cos(radian), 0],
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

    matrix = np.diag(np.ones((3,)))

    matrix = np.dot(matrix, matrix_translation1)
    matrix = np.dot(matrix, matrix_rotate)
    matrix = np.dot(matrix, matrix_scale)
    matrix = np.dot(matrix, matrix_translation2)
    matrix = matrix[:2, :]  # warpAffine只需要2X3的仿射矩阵

    # 使用opencv提供的方法获取仿射矩阵
    matrix2 = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)  # 获取旋转矩阵(旋转中心(pt), 旋转角度(angle)， 缩放系数(scale)
    # assert (matrix - matrix2).sum() < 0e-5, "仿射矩阵不一样"  # 断言一下两种方式生成的仿射矩阵是否一样
    # 进行仿射变换 参数：（输入图像, 2X3的变换矩阵, 指定图像输出尺寸, 插值算法标识符, 边界填充BORDER_REPLICATE)
    dst = cv2.warpAffine(image, matrix, image.shape[:2][::-1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return dst


def affine_matrix_example():
    """二维坐标的仿射变换例子"""
    # 点（4,1）绕点（center_x,center_y）逆时针旋转 angle 度， 然后缩放 scale 倍。
    # 可以分解为：
    # 1.将（center_x,center_y）点平移到原点
    # 2.点（4,1）绕点原点逆时针旋转angle度
    # 3再将得到的坐标乘以缩放比例scale
    # 4.将（center_x,center_y）点平移到回原来的位置
    xy0 = np.array([4, 1, 1]).reshape((3, 1))
    angle = 110
    center_x = 2
    center_y = 1
    scale = 0.5
    # 构建平移矩阵，将中心点从原点移回
    matrix_translation1 = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    # 构建旋转矩阵
    radian = angle * np.pi / 180
    matrix_rotate = np.array([
        [np.cos(radian), -np.sin(radian), 0],
        [np.sin(radian), np.cos(radian), 0],
        [0, 0, 1]
    ])
    # 构建缩放矩阵
    matrix_scale = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    # 构建平移矩阵，将中心点移到原点
    matrix_translation2 = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    # shape (3, 3), 左上角到右下角对角线为1，其余为0的3X3矩阵，该矩阵表示不进行任何平移、旋转及缩放操作
    matrix = np.diag(np.ones((3,)))

    matrix = np.dot(matrix, matrix_translation1)  # 4 中心点从原点移回原来的位置
    matrix = np.dot(matrix, matrix_scale)  # 3 缩放操作
    matrix = np.dot(matrix, matrix_rotate)  # 2 旋转操作
    matrix = np.dot(matrix, matrix_translation2)  # 1 中心点移到原点

    xy1 = np.dot(matrix, xy0)

    print(matrix)
    print(xy0)
    print(xy1)

    # 绘制点的变换前后的位置及中心点位置
    x0, y0 = round(xy0[0, 0], 2), round(xy0[1, 0], 2)  # 点的原始位置
    x1, y1 = round(xy1[0, 0], 2), round(xy1[1, 0], 2)  # 点平移旋转缩放后的位置
    # 定义多条线条
    lines = [[(center_x, center_y), (x0, y0)],  # 中心点到原始位置的直线，开始和结束坐标
             [(center_x, center_y), (x1, y1)]]  # 中心点到变换后的位置的直线，开始和结束坐标
    # 定义线条颜色
    colors = np.array([(1, 0, 0, 1),  # 中心点到原始位置的直线，的颜色RGBA
                       (0, 1, 0, 1)])  # 中心点到变换后的位置的直线，的颜色RGBA
    lc = mc.LineCollection(lines, colors=colors, linewidths=2, linestyles="solid")
    # 开始绘制线条
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    # 显示三个点的坐标
    plt.text(center_x-0.5, center_y-0.5, (center_x, center_y), color='b')
    plt.text(x0, y0, (x0, y0), color='r')
    plt.text(x1, y1, (x1, y1), color='g')
    # 设置横纵坐标轴的最小值和最大值
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.show()


def resize_image_example():
    image = cv2.imread("data/images/img_1001.jpg")
    dst = rotate(image, 80, scale=2)
    cv2.imshow("image", image)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    affine_matrix_example()
    resize_image_example()
