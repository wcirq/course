# -*- coding: utf-8 -*-
# @File train.py
# @Time 2021/3/1 上午9:30
# @Author wcirq
# @Software PyCharm
# @Site 绘制使用不同初始化方法损失的折线图
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets


def model(x, w, b):
    return x@w+b


def train_epoch(epoch, train_dataset, val_dataset, optimizer, w, b):
    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x, w, b)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
            # 训练集测试准确度
            train_correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
            train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, [w, b])
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, [w, b]))
        # 间隔100步计算一次验证集的损失和准确率
        if step % 100 == 0:
            losses = []
            val_accuracies = []
            for index, (x_val, y_val) in enumerate(val_dataset):
                # [b, 28, 28] => [b, 784]
                x_val = tf.reshape(x_val, (-1, 28 * 28))
                # [b, 784] => [b, 10]
                out_val = model(x_val, w, b)
                loss_val = tf.reduce_sum(tf.square(out_val - y)) / x_val.shape[0]
                # 训练集测试准确度
                val_correct_prediction = tf.equal(tf.argmax(out_val, 1), tf.argmax(y_val, 1))
                val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
                losses.append(loss_val.numpy())
                val_accuracies.append(val_accuracy.numpy())
            print(epoch, step, 'loss_train:', loss.numpy(), 'train_accuracy:', train_accuracy.numpy())
            print(epoch, step, 'loss_val:', sum(losses) / len(losses), 'val_accuracy:',
                  sum(val_accuracies) / len(val_accuracies))
            print()
    losses = []
    val_accuracies = []
    for index, (x_val, y_val) in enumerate(val_dataset):
        # [b, 28, 28] => [b, 784]
        x_val = tf.reshape(x_val, (-1, 28 * 28))
        # [b, 784] => [b, 10]
        out_val = model(x_val, w, b)
        loss_val = tf.reduce_sum(tf.square(out_val - y)) / x_val.shape[0]
        # 训练集测试准确度
        val_correct_prediction = tf.equal(tf.argmax(out_val, 1), tf.argmax(y_val, 1))
        val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
        losses.append(loss_val.numpy())
        val_accuracies.append(val_accuracy.numpy())
    print(epoch, step, 'loss_train:', loss.numpy(), 'train_accuracy:', train_accuracy.numpy())
    print(epoch, step, 'loss_val:', sum(losses) / len(losses), 'val_accuracy:',
          sum(val_accuracies) / len(val_accuracies))
    train_loss, val_loss = loss.numpy(), sum(losses) / len(losses)
    train_accuracy, val_accuracy = train_accuracy.numpy(), sum(val_accuracies) / len(val_accuracies)
    return train_loss, val_loss, train_accuracy, val_accuracy


def train(initial_func):
    (x, y), (x_val, y_val) = datasets.mnist.load_data()

    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    print(x.shape, y.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(200)

    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32) / 255.
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
    y_val = tf.one_hot(y_val, depth=10)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(200)

    # 初始化变量
    w = tf.Variable(initial_value=initial_func((784, 10)))
    b = tf.Variable(initial_value=initial_func((10,)))

    optimizer = optimizers.SGD(learning_rate=0.02)
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(20):
        train_loss, val_loss, train_accuracy, val_accuracy = train_epoch(epoch, train_dataset, val_dataset, optimizer, w, b)
        train_losses.append(train_loss)
        val_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    initial_func_list = dict(
        normal=tf.random.normal,
        uniform=tf.random.uniform,
        glorot_normal=tf.initializers.GlorotNormal(),
        glorot_uniform=tf.initializers.GlorotUniform()
    )
    datas = dict(
        train_losses={},
        val_losses={},
        train_accuracies={},
        val_accuracies={},
    )
    for name, func in initial_func_list.items():
        train_losses, val_losses, train_accuracies, val_accuracies = train(func)
        datas["train_losses"][name] = train_losses
        datas["val_losses"][name] = val_losses
        datas["train_accuracies"][name] = train_accuracies
        datas["val_accuracies"][name] = val_accuracies

    for type_name, values in datas.items():
        for initial_func_name, value in values.items():
            plt.plot(value, marker='o', mec='r', mfc='w', label=initial_func_name)
            plt.legend()  # 让图例生效
        plt.show()
