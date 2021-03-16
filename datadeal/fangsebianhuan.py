# -*- coding: utf-8 -*-
# @File fangsebianhuan.py
# @Time 2021/3/17 上午12:06
# @Author wcirq
# @Software PyCharm
# @Site
import numpy as np

if __name__ == '__main__':
    # 点（4,1）绕点（2,1）逆时针旋转45度
    xy0 = np.array([4, 1, 1]).reshape((3, 1))
    angle = 90
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
