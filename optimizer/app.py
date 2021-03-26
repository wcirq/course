# -*- coding: utf-8 -*- 
# @File app.py
# @Time 2021/3/25 15:15
# @Author wcy
# @Software: PyCharm
# @Site
import threading
import time

import tensorflow as tf
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo


class ThreeDChart(object):
    IS_PERSPECTIVE = True  # 透视投影
    VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 0.8, 30.0])  # 视景体的left/right/bottom/top/near/far六个面
    SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
    EYE = np.array([0.1, 0.8, 0.8])  # 眼睛的位置（默认z轴的正方向）
    LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
    EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
    WIN_W, WIN_H = 640, 480  # 保存窗口宽度和高度的变量
    LEFT_IS_DOWNED = False  # 鼠标左键被按下
    MOUSE_X, MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

    def __init__(self, vbo_vertices=None, vbo_indices=None, y_offset_ratio=0.3):
        self.coordinate_scope = 15  # 坐标轴范围
        self.tracks = []  # 历史轨迹坐标[[x1, y1, z1], [x2, y2, z2], ...]
        self.token_tracks = {}  # 多条历史轨迹坐标 {"Adam":[[x1, y1, z1], [x2, y2, z2], ...], ...}
        self.xyz = None  # 当前位置坐标 [w0, loss, w1]
        self.vbo_vertices = vbo_vertices  # 曲面坐标值
        self.vbo_indices = vbo_indices  # 曲面坐标索引
        self.y_offset_ratio = y_offset_ratio  # y轴偏移系数

        self.DIST, self.PHI, self.THETA = self.__getposture()
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_RGBA)

        glutInitWindowSize(self.WIN_W, self.WIN_H)
        glutInitWindowPosition(300, 200)
        glutCreateWindow('梯度下降动画'.encode("gbk"))

        self.__init_canvas()  # 初始化画布
        glutDisplayFunc(self.draw)  # 注册回调函数draw()
        glutTimerFunc(33, self.__move, 1)  # 注册移动回调函数__move()
        glutReshapeFunc(self.__reshape)  # 注册响应窗口改变的函数__reshape()
        glutMouseFunc(self.__mouseclick)  # 注册响应鼠标点击的函数__mouseclick()
        glutMotionFunc(self.__mouse_motion)  # 注册响应鼠标拖拽的函数__mouse_motion()
        glutKeyboardFunc(self.__keydown)  # 注册键盘输入的函数__keydown()

    def update_surface(self, vertices, indices):
        self.vbo_vertices = vbo.VBO(vertices)
        self.vbo_indices = vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER)

    def update_xyz(self, xyz):
        self.xyz = xyz

    def update_token_xyz(self, token, xyz):
        if token in self.token_tracks.keys():
            self.token_tracks[token].append(xyz)
        else:
            self.token_tracks[token] = [xyz]

    def __init_canvas(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）

    def __move(self, args):
        """"""
        glutPostRedisplay()
        glutTimerFunc(33, self.__move, 1)

    def __reshape(self, width, height):
        """窗口改变回调"""
        self.WIN_W, self.WIN_H = width, height
        glutPostRedisplay()

    def __mouseclick(self, button, state, x, y):
        """鼠标点击回调"""
        self.MOUSE_X, self.MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            self.LEFT_IS_DOWNED = state == GLUT_DOWN
        elif button == 3:
            # self.SCALE_K *= 1.05
            self.EYE *= 1.05
            glutPostRedisplay()
        elif button == 4:
            # self.SCALE_K *= 0.95
            self.EYE *= 0.95
            glutPostRedisplay()

    def __getposture(self):
        """眼睛与观察目标之间的距离、仰角、方位角"""
        dist = np.sqrt(np.power((self.EYE - self.LOOK_AT), 2).sum())
        if dist > 0:
            phi = np.arcsin((self.EYE[1] - self.LOOK_AT[1]) / dist)
            theta = np.arcsin((self.EYE[0] - self.LOOK_AT[0]) / (dist * np.cos(phi)))
        else:
            phi = 0.0
            theta = 0.0
        return dist, phi, theta

    def __mouse_motion(self, x, y):
        """鼠标运动回调"""
        if self.LEFT_IS_DOWNED:
            dx = self.MOUSE_X - x
            dy = y - self.MOUSE_Y
            self.MOUSE_X, self.MOUSE_Y = x, y

            self.PHI += 2 * np.pi * dy / self.WIN_H
            self.PHI %= 2 * np.pi
            self.THETA += 2 * np.pi * dx / self.WIN_W
            self.THETA %= 2 * np.pi
            r = self.DIST * np.cos(self.PHI)

            self.EYE[1] = self.DIST * np.sin(self.PHI) * self.coordinate_scope
            self.EYE[0] = r * np.sin(self.THETA) * self.coordinate_scope
            self.EYE[2] = r * np.cos(self.THETA) * self.coordinate_scope

            if 0.5 * np.pi < self.PHI < 1.5 * np.pi:
                self.EYE_UP[1] = -1.0
            else:
                self.EYE_UP[1] = 1.0
            glutPostRedisplay()

    def __keydown(self, key, x, y):
        if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
            if key == b'x':  # 瞄准参考点 x 减小
                self.LOOK_AT[0] -= 0.1
            elif key == b'X':  # 瞄准参考 x 增大
                self.LOOK_AT[0] += 0.1
            elif key == b'y':  # 瞄准参考点 y 减小
                self.LOOK_AT[1] -= 0.1
            elif key == b'Y':  # 瞄准参考点 y 增大
                self.LOOK_AT[1] += 0.1
            elif key == b'z':  # 瞄准参考点 z 减小
                self.LOOK_AT[2] -= 0.1
            elif key == b'Z':  # 瞄准参考点 z 增大
                self.LOOK_AT[2] += 0.1

            self.DIST, self.PHI, self.THETA = self.__getposture()
            glutPostRedisplay()
        elif key == b'\r':  # 回车键，视点前进
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 0.9
            self.DIST, self.PHI, self.THETA = self.__getposture()
            glutPostRedisplay()
        elif key == b'\x08':  # 退格键，视点后退
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 1.1
            self.DIST, self.PHI, self.THETA = self.__getposture()
            glutPostRedisplay()
        elif key == b' ':  # 空格键，切换投影模式
            self.IS_PERSPECTIVE = not self.IS_PERSPECTIVE
            glutPostRedisplay()

    def draw(self):
        """将会一直不断进入该方法绘制图形"""
        # 清除屏幕及深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 设置投影（透视投影）
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.WIN_W > self.WIN_H:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0] * self.WIN_W / self.WIN_H, self.VIEW[1] * self.WIN_W / self.WIN_H, self.VIEW[2],
                          self.VIEW[3], self.VIEW[4],
                          self.VIEW[5])  # glFrustum() 用来设置平行投影
            else:
                glOrtho(self.VIEW[0] * self.WIN_W / self.WIN_H, self.VIEW[1] * self.WIN_W / self.WIN_H, self.VIEW[2],
                        self.VIEW[3], self.VIEW[4],
                        self.VIEW[5])  # glOrtho() 用来设置平行投影
        else:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0], self.VIEW[1], self.VIEW[2] * self.WIN_H / self.WIN_W,
                          self.VIEW[3] * self.WIN_H / self.WIN_W, self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0], self.VIEW[1], self.VIEW[2] * self.WIN_H / self.WIN_W,
                        self.VIEW[3] * self.WIN_H / self.WIN_W, self.VIEW[4], self.VIEW[5])

        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 几何变换
        glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])

        # 设置视点
        gluLookAt(
            self.EYE[0], self.EYE[1], self.EYE[2],
            self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2],
            self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2]
        )

        # 设置视口
        glViewport(0, 0, self.WIN_W, self.WIN_H)

        # ---------------------------------------------------------------
        glLineWidth(1.0)
        glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

        # 以红色绘制x轴
        glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
        glVertex3f(-self.coordinate_scope, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
        glVertex3f(self.coordinate_scope, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）
        glVertex3f(self.coordinate_scope, 0.0, 0.0)  # x轴正方向箭头
        glVertex3f(self.coordinate_scope - 0.1 * self.coordinate_scope, -0.05 * self.coordinate_scope, 0.0)  # x轴正方向箭头

        # 以绿色绘制y轴
        glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
        glVertex3f(0.0, -self.coordinate_scope, 0.0)  # 设置y轴顶点（y轴负方向）
        glVertex3f(0.0, self.coordinate_scope, 0.0)  # 设置y轴顶点（y轴正方向）
        glVertex3f(0.0, self.coordinate_scope, 0.0)  # y轴正方向箭头
        glVertex3f(-0.05 * self.coordinate_scope, self.coordinate_scope - 0.1 * self.coordinate_scope, 0.0)  # y轴正方向箭头

        # 以蓝色绘制z轴
        glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
        glVertex3f(0.0, 0.0, -self.coordinate_scope)  # 设置z轴顶点（z轴负方向）
        glVertex3f(0.0, 0.0, self.coordinate_scope)  # 设置z轴顶点（z轴正方向）
        glVertex3f(0.0, 0.0, self.coordinate_scope)  # z轴正方向箭头
        glVertex3f(0.0, -0.05 * self.coordinate_scope, self.coordinate_scope - 0.1 * self.coordinate_scope)  # z轴正方向箭头

        glEnd()  # 结束绘制线段

        self.start_draw()
        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    def start_draw(self):
        xyz = self.xyz
        if xyz is not None:
            self.tracks.append(xyz)
            # 绘制方形轨迹
            size = 10.0
            glPointSize(size)
            glBegin(GL_POINTS)
            glColor4f(0.0, 1.0, 0.0, 1.0)
            glVertex3f(xyz[0], xyz[1], xyz[2])
            glEnd()

            # 绘制球形轨迹
            # glPushMatrix()
            # glTranslatef(xyz[0], xyz[1], xyz[2])
            # glColor4f(0, 1, 1, 1)
            # glutWireSphere(scope * 0.08, 10, 10)
            # # glutSolidSphere(scope*0.08, 10, 10)
            # glPopMatrix()

        if self.vbo_vertices is not None and self.vbo_indices is not None:
            # 绘制曲面
            self.vbo_vertices.bind()
            # glInterleavedArrays(GL_V3F, 0, None)
            glInterleavedArrays(GL_C3F_V3F, 0, None)
            self.vbo_vertices.unbind()
            self.vbo_indices.bind()
            # glDrawElements(GL_QUADS, int(vbo_indices.size / 4), GL_UNSIGNED_INT, None)
            glDrawElements(GL_TRIANGLES, int(self.vbo_indices.size / 4), GL_UNSIGNED_INT, None)
            self.vbo_indices.unbind()
        if len(self.tracks) > 0:
            # 绘制单条轨迹
            glEnable(GL_BLEND)
            # glEnable(GL_LINE_SMOOTH)
            glLineWidth(2.0)
            glBegin(GL_LINES)  # 开始绘制线段
            glColor4f(0.0, 0.9, 0.0, 1.0)  # 设置当前颜色为红色不透明
            for i in range(len(self.tracks) - 1):
                glVertex3f(self.tracks[i][0], float(self.tracks[i][1]), self.tracks[i][2])
                glVertex3f(self.tracks[i + 1][0], float(self.tracks[i + 1][1]), self.tracks[i + 1][2])
            glEnd()
        if len(self.token_tracks.keys()) > 0:
            # 绘制多条轨迹
            np.random.seed(0)  # 设置随机种子，使每次重新迭代生成的轨迹颜色一致
            for index, (key_name, values) in enumerate(self.token_tracks.items()):
                # 随机为轨迹生成颜色
                rgb = np.random.random((3,))
                # 绘制轨迹
                glEnable(GL_BLEND)
                # glEnable(GL_LINE_SMOOTH)
                glLineWidth(2.0)
                glBegin(GL_LINES)  # 开始绘制线段
                glColor4f(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)  # 设置当前颜色为红色不透明
                for i in range(len(values) - 1):
                    # y坐标稍稍加上y_offset，使多条线分离，方便查看
                    y_offset = index*self.y_offset_ratio
                    glVertex3f(values[i][0], float(values[i][1]+y_offset), values[i][2])
                    glVertex3f(values[i + 1][0], float(values[i + 1][1]+y_offset), values[i + 1][2])
                glEnd()

    def loop(self):
        glutMainLoop()


class Linear(tf.keras.layers.Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(2, 1),
                                 initializer=tf.random.normal,
                                 trainable=True,
                                 name="w")

    def call(self, inputs, training=None, mask=None):
        return tf.matmul(inputs, self.w)


class Model(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(args, kwargs)
        self.liner = Linear()

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch_size, 2]
        outputs = self.liner(inputs)
        outputs = tf.reshape(outputs, (-1,))
        return outputs


class MeanSquaredLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))


def get_dataset(w=None, num=100, batch_size=10, plt=False):
    """
    生成数据
    :param num: 数据量
    :param batch_size: 批处理大小
    :param plt: 是否绘制散点图
    :return:
    """
    if w is None:
        w = [0.2, 0.2]
    x_data = np.random.random((num, 2)) * 2 - 1
    x_data = x_data.astype(np.float32)
    y_data = np.zeros((num,), dtype=np.float32)
    y = np.dot(x_data, np.array(w, dtype=np.float32))
    loc = y < y.mean()
    y_data[loc] = 1
    if plt:
        import matplotlib.pyplot as plt
        x1, y1 = x_data[loc, 0], x_data[loc, 1]
        x2, y2 = x_data[~loc, 0], x_data[~loc, 1]
        plt.scatter(x1, y1, c='#00CED1', alpha=0.8, label='类别A')
        plt.scatter(x2, y2, c='#DC143C', alpha=0.8, label='类别B')
        plt.show()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_data, y))
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset, x_data, y


# RGB格式颜色转换为16进制颜色格式
def RGB_to_Hex(rgb):
    RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


# RGB格式颜色转换为16进制颜色格式
def RGB_list_to_Hex(RGB):
    # RGB = rgb.split(',')  # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


# 16进制颜色格式颜色转换为RGB格式
def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = str(r) + ',' + str(g) + ',' + str(b)
    return rgb, [r, g, b]


# 生成渐变色
def gradient_color(color_list, color_sum=200):
    color_center_count = len(color_list)
    # if color_center_count == 2:
    #     color_center_count = 1
    color_sub_count = int(color_sum / (color_center_count - 1))
    color_index_start = 0
    color_map = []
    for color_index_end in range(1, color_center_count):
        color_rgb_start = Hex_to_RGB(color_list[color_index_start])[1]
        color_rgb_end = Hex_to_RGB(color_list[color_index_end])[1]
        r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
        g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
        b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
        # 生成中间渐变色
        now_color = color_rgb_start
        # color_map.append(RGB_list_to_Hex(now_color))
        color_map.append(now_color)
        for color_index in range(1, color_sub_count):
            now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
            # color_map.append(RGB_list_to_Hex(now_color))
            color_map.append(now_color)
        color_index_start = color_index_end
    return color_map


def getColor(scale, input_colors=None):
    if input_colors is None:
        input_colors = ["#b93f43", "#882288", "#037ee4"]
    colors = gradient_color(input_colors, color_sum=300)
    colors = np.array(colors)[:255, :] / 255
    R = colors[:, 0]
    G = colors[:, 1]
    B = colors[:, 2]
    A = np.ones(255, dtype=np.float) * 0.5
    scale = min(scale, 254)
    return R[scale], B[scale], B[scale], A[scale]


def deal_vbo(w0, w1, loss):
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
    return np.array(vertices, np.float32), np.array(indices, np.int)


def build_surface(model: Model, loss_fun: MeanSquaredLoss, x_data, y_data, init_w=None):
    num = 50
    if init_w is None:
        min_w = -4
        max_w = 4
    else:
        min_w = -init_w.max() * 1.1
        max_w = init_w.max() * 1.1
    ws0 = np.linspace(min_w, max_w, num)
    ws1 = np.linspace(min_w, max_w, num)
    losses = np.zeros((ws0.shape[0], ws1.shape[0]), dtype=np.float32)
    # w = tf.identity(model.trainable_weights[0])  # 赋值一份一样的Tensor,且重新分配内存
    for i, w0 in enumerate(ws0):
        for j, w1 in enumerate(ws1):
            model.trainable_weights[0].assign(tf.convert_to_tensor(np.array([[w0], [w1]], dtype=np.float32)))
            y_predict = model(x_data)
            loss = loss_fun(y_data, y_predict)
            losses[i, j] = loss
    vertices, indices = deal_vbo(ws0, ws1, losses)
    return vertices, indices


def run(tdc: ThreeDChart, optimizer, train_dataset):
    init_w = np.array([[1], [5]], dtype=np.float32)
    model = Model()
    loss_fun = MeanSquaredLoss()
    # 给要训练的变量赋新值
    model.trainable_weights[0].assign(init_w)
    while True:
        for step, (x, y_) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y = model(x)
                loss = loss_fun(y_, y)
                w0, w1 = model.trainable_weights[0][:, 0].numpy()
                xyz = [w0, loss.numpy(), w1]
                tdc.update_token_xyz(optimizer._name, xyz)
                # print("loss", loss.numpy(), [i.numpy() for i in model.trainable_weights])
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))


def train(tdc: ThreeDChart):
    num = 500
    batch_size = 500
    init_w = np.array([[1], [5]], dtype=np.float32)
    train_dataset, x_data, y_data = get_dataset(num=num, batch_size=batch_size)
    model = Model()
    loss_fun = MeanSquaredLoss()
    vertices, indices = build_surface(model, loss_fun, x_data, y_data, init_w=init_w)
    tdc.update_surface(vertices, indices)
    # 给要训练的变量赋新值
    model.trainable_weights[0].assign(init_w)
    optimizers = [
        tf.optimizers.SGD(learning_rate=0.002),  # 随机梯度下降
        tf.optimizers.Nadam(learning_rate=0.002),  # 随机梯度下降
        # tf.optimizers.Nadam(learning_rate=0.002),  # 随机梯度下降
        tf.optimizers.Adagrad(learning_rate=0.002),  # 随机梯度下降
        # tf.optimizers.Adadelta(learning_rate=0.002),  # 随机梯度下降
        tf.optimizers.Adam(learning_rate=0.002),  # 随机梯度下降
    ]

    # for optimizer in optimizers:
    #     threading.Thread(target=run, args=(tdc, optimizer, train_dataset)).start()

    optimizer = optimizers[3]
    epoch = 1000
    start = time.time()
    for e in range(epoch):
        for step, (x, y_) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y = model(x)
                loss = loss_fun(y_, y)
                w0, w1 = model.trainable_weights[0][:, 0].numpy()
                xyz = [w0, loss.numpy(), w1]
                tdc.update_xyz(xyz)
                # print("loss", loss.numpy(), [i.numpy() for i in model.trainable_weights])
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print(optimizer._name, "time", time.time()-start, "loss", loss.numpy())


if __name__ == '__main__':
    tdc = ThreeDChart(y_offset_ratio=0.3)
    threading.Thread(target=train, args=(tdc,)).start()
    tdc.loop()
