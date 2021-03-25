# -*- coding: utf-8 -*- 
# @File main.py
# @Time 2021/3/22 10:10
# @Author wcy
# @Software: PyCharm
# @Site
import os
import random
import cv2
import numpy as np
from tqdm  import tqdm
random.seed(206211)


def encode(img, watermark, alpha=0.5):
    img_f = np.fft.fft2(img)
    # cv2.imshow("", watermark)
    # cv2.waitKey(0)
    abs_img_f = np.abs(img_f)
    watermark = abs_img_f.max()/watermark.max()*watermark*0.5
    res_f = img_f + alpha * watermark
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    res = np.clip(res, a_min=0, a_max=255)
    res = res.astype(np.uint8)
    return res


def decode(ori, img, alpha=0.5):
    ori_f = np.fft.fft2(ori)
    img_f = np.fft.fft2(img)
    height, width = ori.shape[0], ori.shape[1]
    watermark = (img_f - ori_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(watermark.shape)
    # random.seed(height + width)
    x = list(range(height))
    y = list(range(width))
    # random.shuffle(x)
    # random.shuffle(y)
    for i in range(height):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]
    res = (res - res.min()) / (res.max() - res.min())
    res = res * 255
    res = res.astype(np.uint8)
    return res


def main():
    root = r"E:\DATA\安全AI挑战者计划第五期\ImageNet无限制对抗攻击\imagenet_round1_210122\images\images"
    save = r"E:\DATA\安全AI挑战者计划第五期\ImageNet无限制对抗攻击\imagenet_round1_210122\results\images"
    images_files = os.listdir(root)
    for file1 in tqdm(images_files[2380:]):
        for file2 in images_files:
            if file1 == file2:
                continue
            image_path1 = os.path.join(root, file1)
            image_path2 = os.path.join(root, file2)
            image = cv2.imdecode(np.fromfile(image_path1, dtype=np.uint8), -1)
            watermark = cv2.imdecode(np.fromfile(image_path2, dtype=np.uint8), -1)
            frame = encode(image, watermark, alpha=0.9)
            decode_frame = decode(image, frame, alpha=0.9)
            cv2.imencode('.jpg', frame)[1].tofile(os.path.join(save, file1))
            # cv2.imshow("frame", np.hstack((image, frame, decode_frame)))
            # cv2.waitKey(0)
            break


if __name__ == '__main__':
    main()
