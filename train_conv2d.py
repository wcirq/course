# -*- coding: utf-8 -*-
# @File train.py
# @Time 2021/3/1 上午9:30
# @Author wcirq
# @Software PyCharm
# @Site
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets


def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)


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

model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Reshape((-1,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)]
    )

optimizer = optimizers.SGD(learning_rate=0.01)


def train_epoch(epoch):
    # Step4.循环迭代
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 28, 28, 1]
            x = tf.reshape(x, (-1, 28, 28, 1))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
            # 训练集测试准确度
            train_correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
            train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

        # Step3. 优化更新 w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 间隔100步计算一次验证集的损失和准确率
        if step % 100 == 0:
            losses = []
            val_accuracies = []
            for index, (x_val, y_val) in enumerate(val_dataset):
                # [b, 28, 28] => [b, 784]
                x_val = tf.reshape(x_val, (-1, 28, 28, 1))
                # Step1. 计算输出
                # [b, 784] => [b, 10]
                out_val = model(x_val)
                loss_val = tf.reduce_sum(tf.square(out_val - y)) / x_val.shape[0]
                # 训练集测试准确度
                val_correct_prediction = tf.equal(tf.argmax(out_val, 1), tf.argmax(y_val, 1))
                val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
                losses.append(loss_val.numpy())
                val_accuracies.append(val_accuracy.numpy())
            print(epoch, step, 'loss_train:', loss.numpy(), 'train_accuracy:', train_accuracy.numpy())
            print(epoch, step, 'loss_val:', sum(losses)/len(losses), 'val_accuracy:', sum(val_accuracies)/len(val_accuracies))
            print()


def train():
    for epoch in range(30):
        train_epoch(epoch)
    model.save("resources/model/model_conv2d.h5")


if __name__ == '__main__':
    train()