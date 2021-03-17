# -*- coding: utf-8 -*- 
# @File gaussianblur.py
# @Time 2021/3/16 20:18
# @Author wcy
# @Software: PyCharm
# @Site
import cv2
import numpy as np


def gaussian_kernel(shape, std=1):
    h, w = shape[0], shape[1]
    kernel_index_h = np.linspace(-h // 2 + 1, h // 2, h)
    kernel_index_w = np.linspace(-w // 2 + 1, w // 2, w)
    x, y = np.meshgrid(kernel_index_h, kernel_index_w)
    kernel = np.exp(-np.square(np.square(x) + np.square(y)) / (2 * np.square(std))) / (2 * np.pi * np.square(std))
    kernel = kernel / kernel.sum()
    return kernel


def my_gaussian_blur(img, shape, std=1):
    h, w = shape
    assert len(img.shape) == 2, "只支持单通道图片"
    assert h % 2 == 1 and w % 2 == 1, "shape的值必须为奇数"
    # 归一化
    img = (img-img.min())/(img.max()-img.min())
    # 定义卷积核
    kernel = gaussian_kernel((h, w), std=std)
    # 计算上下左右填充的数据行/列数
    top = (h - 1) // 2
    bottom = (h - 1) // 2
    left = (w - 1) // 2
    right = (w - 1) // 2
    img_pad = np.pad(img, ((top, bottom), (left, right)), 'constant', constant_values=(0, 0))
    # 定义一个与 img shape一致的矩阵用于存储卷积运算结果
    result = np.zeros_like(img, dtype=np.float64)
    img_h, img_w = img.shape
    for i in range(img_h):
        for j in range(img_w):
            result[i, j] = (img_pad[i:i + h, j:j+w] * kernel).sum()
    return (result*255).astype(np.uint8)


def main():
    std = 1
    h, w = 5, 5
    # img = cv2.imread("data/images/img_1001.jpg", 0)[:50, :50]
    img = np.diag(np.ones((50,)))
    res1 = cv2.GaussianBlur(img, (h, w), std)
    res2 = my_gaussian_blur(img, (h, w), std=std)
    cv2.imshow("img", img)
    cv2.imshow("res1", res1)
    cv2.imshow("res2", res2)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
