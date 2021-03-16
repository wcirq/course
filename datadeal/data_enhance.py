# -*- coding: utf-8 -*- 
# @File data_enhance.py
# @Time 2021/3/16 15:15
# @Author wcy
# @Software: PyCharm
# @Site 
import cv2
import numpy as np


def get_noise(shape, rate=0.001):
    """
    生成噪声图像
    :param shape:生成的噪声矩阵shape
    :param rate: 噪声数占总像素的比例
    :return:
    """
    # 随机生成不同强度的噪点
    noise = np.random.uniform(0, 256, shape)
    min_value = (1-rate) * 256
    # 将小于separate的去除
    noise[np.where(noise < min_value)] = 0
    print(noise[noise>0].size/(shape[0]*shape[1]))
    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    # 可以输出噪声看看
    # cv2.imshow('img',noise)
    # cv2.waitKey()
    # cv2.destroyWindow('img')
    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    """
    将噪声加上运动模糊,模仿雨滴
    :param noise: 输入噪声图
    :param length: 对角矩阵大小，表示雨滴的长度
    :param angle: 倾斜的角度，逆时针为正
    :param w: 雨滴大小
    :return: 输出带模糊的噪声
    """
    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对角矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度
    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波
    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = blurred.astype(np.uint8)
    # 看结果
    # cv2.imshow('img',blurred)
    # cv2.waitKey()
    # cv2.destroyWindow('img')

    return blurred


def alpha_rain(rain, img, beta=0.8):
    """
    在图片上合成雨滴
    :param rain: 雨滴图
    :param img: 原始图片
    :param beta: 雨滴的权重
    :return:
    """
    rain = np.expand_dims(rain, 2)
    rain_result = img.copy()  # 拷贝一个份
    rain = rain.astype(np.float32)
    # (255 - rain[:, :, 0]) / 255.0 的结果是0-1之间的值，结果接近0的部分表示雨滴，结果接近1的部分表示背景
    # 0(雨滴部分)乘以原图的第一通道(rain_result[:, :, 0])的对应像素值，原图上该位置值变为0，然后0加上 beta * rain[:, :, 0](此处为表示雨滴的值), 即在原图上得到雨滴
    # 1(背景部分)乘以原图的第一通道(rain_result[:, :, 0])的对应像素值，原图上该位置值不变，然后该值加上 beta * rain[:, :, 0](此处为表示背景的值-0), 即在原图上保留背景
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    cv2.imshow('rain_effct_result', rain_result)
    cv2.waitKey(0)
    cv2.destroyWindow("rain_effct_result")


def main():
    image = cv2.imread(r"data/images/img_1001.jpg")
    noise = get_noise(image.shape[0:2], rate=0.05)
    rain_noise = rain_blur(noise, length=20, angle=45, w=1)
    alpha_rain(rain_noise, image)


if __name__ == '__main__':
    main()