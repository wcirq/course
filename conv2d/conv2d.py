"""
利用卷积核重排实现了卷积运算
"""
import numpy as np


def k_reset(k, x, s=1, p=0):
    if p > 0:
        x = np.pad(x, (p, p))
    x_h, x_w = x.shape
    k_h, k_w = k.shape
    s_h = s_w = s
    p_h = p_w = 0
    o_h = int((x_h - k_h + 2 * p_h) / s_h + 1)
    o_w = int((x_w - k_w + 2 * p_w) / s_w + 1)
    KR = np.zeros((o_h * o_w, x.size), dtype=np.float32)
    for i in range(o_w):
        for j in range(o_h):
            temp_kr = np.zeros_like(x)
            temp_kr[j * s_h:j * s_h + k_h, i * s_w:i * s_w + k_w] = k
            KR[j * o_w + i] = temp_kr.flatten()
    return KR, (o_h, o_w)


def x_reset(x, p=0):
    if p > 0:
        x = np.pad(x, (p, p))
    return x.reshape((x.size, 1))


def y_reset(y, o_h, o_w):
    return y.reshape((o_h, o_w))


def conv2d(x, k, s=1, p=0):
    kr, (o_h, o_w) = k_reset(k, x, s=s, p=p)
    xr = x_reset(x, p=p)
    yr = np.dot(kr, xr)
    y = y_reset(yr, o_h, o_w)
    return y


def main():
    s = 2
    p = 1
    x = (np.random.random((5, 5))*10).astype(np.int32)
    # x = np.arange(1, 10).reshape((3, 3))
    k = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y = conv2d(x, k, s=s, p=p)
    print(x, end="\n\n")
    print(k, end="\n\n")
    print(y, end="\n\n")


if __name__ == '__main__':
    main()
