# -*- coding: utf-8 -*- 
# @File main.py
# @Time 2021/3/25 13:47
# @Author wcy
# @Software: PyCharm
# @Site
import copy
import random
import time
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import optimizer.autodiff as ad
import numpy as np
import threading
from optimizer.gradient import gradient_color

IS_PERSPECTIVE = True  # 透视投影
VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 0.5, 50.0])  # 视景体的left/right/bottom/top/near/far六个面
SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
EYE = np.array([0.0, 1.0, 1.0])  # 眼睛的位置（默认z轴的正方向）
LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
WIN_W, WIN_H = 640, 480  # 保存窗口宽度和高度的变量
LEFT_IS_DOWNED = False  # 鼠标左键被按下
MOUSE_X, MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

input_colors = ["#b93f43", "#882288", "#037ee4"]
colors = gradient_color(input_colors, color_sum=300)
colors = np.array(colors)[:255, :] / 255
R = colors[:, 0]
G = colors[:, 1]
B = colors[:, 2]

# R = np.hstack((np.linspace(200, 255, 127), np.linspace(255, 47, 128)))[::-1] / 255
# G = np.hstack((np.linspace(3, 255, 127), np.linspace(255, 47, 128)))[::-1] / 255
# B = np.hstack((np.linspace(33, 200, 127), np.hstack((np.linspace(220, 255, 25), np.linspace(255, 150, 103)))))[
#     ::-1] / 255
A = np.ones(255, dtype=np.float) * 0.5
TAG_XYZ = [0.0, -0.2, 0.0]
loss_mean, loss_std = None, None
tracks = []  # 轨迹


def getposture():
    global EYE, LOOK_AT

    dist = np.sqrt(np.power((EYE - LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1] - LOOK_AT[1]) / dist)
        theta = np.arcsin((EYE[0] - LOOK_AT[0]) / (dist * np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0

    return dist, phi, theta


DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）


def getColor(scale):
    global R, G, B, A
    scale = min(scale, 254)
    return R[scale], B[scale], B[scale], A[scale]


def start_draw():
    global w0, w1, loss, TAG_XYZ, vbo_vertices, vbo_indices, tracks, scope
    XYZ = copy.deepcopy(TAG_XYZ)
    tracks.append(XYZ)

    # 绘制方形轨迹
    size = 10.0
    glPointSize(size)
    glBegin(GL_POINTS)
    glColor4f(0.0, 1.0, 0.0, 1.0)
    # glVertex3f(XYZ[0], XYZ[1] + size / 5, XYZ[2])
    glVertex3f(XYZ[0], XYZ[1], XYZ[2])
    glEnd()

    # 绘制球形轨迹
    # glPushMatrix()
    # glTranslatef(XYZ[0], XYZ[1], XYZ[2])
    # glColor4f(0, 1, 1, 1)
    # glutWireSphere(scope * 0.08, 10, 10)
    # # glutSolidSphere(scope*0.08, 10, 10)
    # glPopMatrix()

    # 绘制曲面
    vbo_vertices.bind()
    # glInterleavedArrays(GL_V3F, 0, None)
    glInterleavedArrays(GL_C3F_V3F, 0, None)
    vbo_vertices.unbind()
    vbo_indices.bind()
    # glDrawElements(GL_QUADS, int(vbo_indices.size / 4), GL_UNSIGNED_INT, None)
    glDrawElements(GL_TRIANGLES, int(vbo_indices.size / 4), GL_UNSIGNED_INT, None)
    vbo_indices.unbind()

    # 绘制轨迹
    glEnable(GL_BLEND)
    # glEnable(GL_LINE_SMOOTH)
    glLineWidth(2.0)
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）
    glColor4f(0.9, 0.9, 0.9, 1.0)  # 设置当前颜色为红色不透明
    for i in range(len(tracks) - 1):
        glVertex3f(tracks[i][0], float(tracks[i][1]), tracks[i][2])
        glVertex3f(tracks[i + 1][0], float(tracks[i + 1][1]), tracks[i + 1][2])
    glEnd()


def draw():
    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H
    global scope

    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4],
                      VIEW[5])  # glFrustum() 用来设置平行投影
        else:
            glOrtho(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4],
                    VIEW[5])  # glOrtho() 用来设置平行投影
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

    # ---------------------------------------------------------------
    glLineWidth(1.0)
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glVertex3f(-scope, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(scope, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）
    glVertex3f(scope, 0.0, 0.0)  # x轴正方向箭头
    glVertex3f(scope - 0.1 * scope, -0.05 * scope, 0.0)  # x轴正方向箭头

    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -scope, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, scope, 0.0)  # 设置y轴顶点（y轴正方向）
    glVertex3f(0.0, scope, 0.0)  # y轴正方向箭头
    glVertex3f(-0.05 * scope, scope - 0.1 * scope, 0.0)  # y轴正方向箭头

    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -scope)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, scope)  # 设置z轴顶点（z轴正方向）
    glVertex3f(0.0, 0.0, scope)  # z轴正方向箭头
    glVertex3f(0.0, -0.05 * scope, scope - 0.1 * scope)  # z轴正方向箭头

    glEnd()  # 结束绘制线段

    start_draw()
    # ---------------------------------------------------------------
    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


def reshape(width, height):
    global WIN_W, WIN_H

    WIN_W, WIN_H = width, height
    glutPostRedisplay()


def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y

    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()


def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H

    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y

        PHI += 2 * np.pi * dy / WIN_H
        PHI %= 2 * np.pi
        THETA += 2 * np.pi * dx / WIN_W
        THETA %= 2 * np.pi
        r = DIST * np.cos(PHI)

        EYE[1] = DIST * np.sin(PHI) * scope
        EYE[0] = r * np.sin(THETA) * scope
        EYE[2] = r * np.cos(THETA) * scope

        if 0.5 * np.pi < PHI < 1.5 * np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0
        glutPostRedisplay()


def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW

    if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
        if key == b'x':  # 瞄准参考点 x 减小
            LOOK_AT[0] -= 0.1
        elif key == b'X':  # 瞄准参考 x 增大
            LOOK_AT[0] += 0.1
        elif key == b'y':  # 瞄准参考点 y 减小
            LOOK_AT[1] -= 0.1
        elif key == b'Y':  # 瞄准参考点 y 增大
            LOOK_AT[1] += 0.1
        elif key == b'z':  # 瞄准参考点 z 减小
            LOOK_AT[2] -= 0.1
        elif key == b'Z':  # 瞄准参考点 z 增大
            LOOK_AT[2] += 0.1

        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\r':  # 回车键，视点前进
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\x08':  # 退格键，视点后退
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b' ':  # 空格键，切换投影模式
        IS_PERSPECTIVE = not IS_PERSPECTIVE
        glutPostRedisplay()


def logistic_prob(_w):
    def wrapper(_x):
        return 1 / (1 + np.exp(-np.sum(_x * _w)))

    return wrapper


def test_accuracy(_w, _X, _Y):
    prob = logistic_prob(_w)
    correct = 0
    total = len(_Y)
    for i in range(len(_Y)):
        x = _X[i]
        y = _Y[i]
        p = prob(x)
        if p >= 0.5 and y == 1.0:
            correct += 1
        elif p < 0.5 and y == 0.0:
            correct += 1
    print("总数：%d, 预测正确：%d" % (total, correct))


def gen_2d_data(n):
    np.random.seed(3)
    x_data = np.random.random([n, 1])
    y_data = np.ones(n)
    for i in range(n):
        d = x_data[i]
        if d[0] < 0.5:
            y_data[i] = 0
    x_data_with_bias = np.ones([n, 2])
    x_data_with_bias[:, 1:] = x_data
    return x_data_with_bias, y_data


def gen_2d_data2(n):
    np.random.seed()
    x_data = np.random.random([n, 2]) * 4 - 2

    w = np.random.random_integers(-5, 5, [2, 1])/1.0
    b = np.random.random_integers(-100, 100, [1])/100
    print(f"W = {w} b = {b}")
    hidden = np.matmul(x_data, w)[:, 0] + b
    y_data = 1 / (1 + np.exp(-hidden))
    #
    # y_data = np.where(np.sum(x_data, axis=1) > 0, 0, 1)

    # y_data = 1 - np.round(np.mean(np.square(x_data), axis=1), decimals=0)
    return x_data, y_data


def gen_2d_data3(n):
    np.random.seed()
    x_data1 = np.random.rand(int(n/2), 2)*0.2+0.2
    x_data2 = np.random.rand(int(n/2), 2)*0.2+0.7
    x_data = np.concatenate((x_data1, x_data2))

    y_data1 = np.zeros_like(x_data1[:, 0])
    y_data2 = np.ones_like(x_data1[:, 0])
    y_data = np.concatenate((y_data1, y_data2))

    c = list(zip(x_data, y_data))
    np.random.shuffle(c)
    x_data, y_data = zip(*c)
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


def gen_2d_data4(n):
    w = np.random.random_integers(-10, 10, [2, 1])/2.0
    x_data = np.random.random([n, 2]) * 2 - 1
    y_data = np.cos(np.sin(np.matmul(x_data, w))).flatten()
    return x_data, y_data


def cost(theta0, theta1):
    global data_x, data_y
    J = 0
    m = len(data_x)
    hidden = np.dot(data_x, np.array([theta0, theta1]))
    sigmoid = 1 / (1 + np.exp(-hidden))
    #
    # hidden = np.cos(np.sin(np.matmul(data_x, np.array([theta0, theta1])))).flatten()
    # sigmoid = hidden
    J = np.sum(np.square(sigmoid - data_y))
    # J = np.sum(-data_y * ad.log(sigmoid+0.01) - (1 - data_y) * ad.log(1 - sigmoid+0.01))
    return J


def move(args):
    glutPostRedisplay()
    glutTimerFunc(33, move, 1)


def train():
    global X_val, Y_val, w_val, TAG_XYZ, N, loss, loss_mean, loss_std
    x = ad.Variable(name='x')
    w = ad.Variable(name='w')
    y = ad.Variable(name='y')

    # 注意，以下实现某些情况会有很大的数值误差，
    # 所以一般真实系统实现会提供高阶算子，从而减少数值误差

    hidden = ad.matmul(x, w)
    # hidden = ad.cos(ad.sin(ad.matmul(x, w)))
    # sigmoid = hidden
    sigmoid = 1 / (1 + ad.exp(-hidden))
    # L = -y * ad.log(sigmoid+0.01) - (1 - y) * ad.log(1 - sigmoid+0.01)
    L = ad.reduce_sum(ad.square(sigmoid - y))

    w_grad, = ad.gradients(L, [w])
    executor = ad.Executor([L, sigmoid, w_grad])

    test_accuracy(w_val, X_val, Y_val)
    batch = 10
    learning_rate = 0.6
    max_iters = 5000
    for iteration in range(max_iters):
        for i in range(int(N / batch)):
            x_val = X_val[i * batch:i * batch + batch, :]
            y_val = Y_val[i * batch:i * batch + batch]
            L_val, predict, w_grad_val = executor.run(feed_dict={w: w_val, x: x_val, y: y_val})
            w_val -= learning_rate * w_grad_val
            TAG_XYZ[0] = float(w_val[0])
            TAG_XYZ[2] = float(w_val[1])
            l = cost(float(w_val[0]), float(w_val[1]))
            TAG_XYZ[1] = (l - loss_mean) / loss_std * 5
            print(f"iter = {iteration}, w = {w_val}, loss={l}")
            time.sleep(0.01)
    test_accuracy(w_val, X_val, Y_val)


def deal_vbo(w0, w1, loss):
    s = time.time()
    vertices = []
    indices = []
    loss_flatten = loss.flatten()
    loss_flatten.sort()
    edge_max = loss_flatten[-20]
    edge_min = loss_flatten[20]
    for i in range(len(w0)):
        for j in range(len(w1)):
            point = [float(w0[i]), float(loss[i][j]), float(w1[j])]
            rgba = getColor(int((float(loss[i][j]) - loss.min()) / (loss.max() - loss.min()) * 255))
            if float(loss[i][j]) < edge_min:
                rgba = (1.0, 0.0, 0.0, 0.5)
            if float(loss[i][j]) > edge_max:
                rgba = (0.0, 1.0, 0.0, 0.5)
            vertices += rgba[:-1]
            vertices += point
            if i < len(w0) - 1 and j < len(w1) - 1:
                # indices += [i * len(w0) + j, (i + 1) * len(w0) + j, (i + 1) * len(w0) + j + 1, i * len(w0) + j + 1]
                indices += [i * len(w0) + j,
                            (i + 1) * len(w0) + j,
                            (i + 1) * len(w0) + j + 1
                            ]
                indices += [i * len(w0) + j,
                            (i + 1) * len(w0) + j + 1,
                            i * len(w0) + j + 1
                            ]
    print("ddd", time.time() - s)
    return np.array(vertices, np.float32), np.array(indices, np.int)


if __name__ == "__main__":
    N = 100
    # X_val, Y_val = gen_2d_data(N)
    X_val, Y_val = gen_2d_data2(N)
    # X_val, Y_val = gen_2d_data3(N)
    # X_val, Y_val = gen_2d_data4(N)
    data_x = X_val
    data_y = Y_val
    # w_val = np.zeros(2)
    # w_val = np.array([-0.0, -1.0])
    w_val = np.random.random_integers(-10, 10, [2])/1.0  # 初始化权重
    print(w_val)

    scope = 15
    EYE *= scope
    SCALE_K *= 1 / scope
    num = 200
    w0 = np.linspace(-scope, scope, num)
    w1 = np.linspace(-scope, scope, num)
    loss = np.empty(shape=(num, num))
    # w0, w1 = np.meshgrid(w0, w1)
    for i in range(num):
        for j in range(num):
            loss[i, j] = cost(w0[i], w1[j])
    loss_mean = loss.mean()
    loss_std = loss.std()
    loss = (loss - loss_mean) / loss_std * 5

    vertices, indices = deal_vbo(w0, w1, loss)

    vbo_vertices = vbo.VBO(vertices)
    vbo_indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_RGBA)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('梯度下降动画'.encode("gbk"))

    init()  # 初始化画布
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutTimerFunc(33, move, 1)
    glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
    glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

    t = threading.Thread(target=train, args=())
    t.start()

    glutMainLoop()  # 进入glut主循环
