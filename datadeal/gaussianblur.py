# -*- coding: utf-8 -*- 
# @File gaussianblur.py
# @Time 2021/3/16 20:18
# @Author wcy
# @Software: PyCharm
# @Site
import cv2
import numpy as np


def normal(x, y=None, std=1.):
    if y is None:
        y = np.exp(-np.square(x) / (2 * np.square(std))) / (std * np.sqrt(2 * np.pi))
    else:
        y = np.exp(-np.square(np.square(x) + np.square(y)) / (2 * np.square(std))) / (2 * np.pi * np.square(std))
    return y


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
    kernel = gaussian_kernel((h, w), std=std)
    top = 1
    bottom = 1
    left = 1
    right = 1
    img_pad = np.pad(img, ((top, bottom), (left, right)), 'constant', constant_values=(0, 0))
    return


def main():
    x = np.array([-1, 0, 1])
    y = normal(x, std=1.5)
    std = 1
    h, w = 3, 3
    # img = np.ones((h, w), dtype=np.float32)
    img = np.random.random((h, w))
    res1 = cv2.GaussianBlur(img, (h, w), std)
    res2 = my_gaussian_blur(img, (h, w), std=std)
    print()


if __name__ == '__main__':
    main()
