# -*- coding: utf-8 -*-
# @File train_tensorboard.py
# @Time 2021/3/9 19:28
# @Author wcy
# @Software: PyCharm
# @Site
import tensorflow as tf
import datetime
from tensorflow.keras import layers


class Argmax(layers.Layer):
    def __init__(self):
        super(Argmax, self).__init__()

    def build(self, axis=-1):
        self.axis = axis

    def call(self, input, axis=-1):
        if axis is not None:
            self.axis = axis
        output = tf.argmax(input, axis=self.axis)
        return output


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)


# def create_model():
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(28, 28)),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])

def create_model():
    layers = [tf.keras.layers.Flatten(input_shape=(28, 28))]
    layers.extend([tf.keras.layers.Dense(32, activation='tanh', kernel_initializer="random_normal") for i in range(10)])
    # layers.extend([tf.keras.layers.Dense(32, activation='tanh', kernel_initializer="glorot_normal") for i in range(10)])
    # layers.append(tf.keras.layers.Dense(10, activation='softmax'))
    layers.append(tf.keras.layers.Dense(10))
    return tf.keras.models.Sequential(layers)


model = create_model()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(60000).batch(200)
test_dataset = test_dataset.batch(200)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
loss_object = tf.keras.losses.MSE
# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)


train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    y_train = tf.argmax(y_train, axis=-1)
    train_accuracy(y_train, predictions)


def test_step(x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    y_test = tf.argmax(y_test, axis=-1)
    test_accuracy(y_test, predictions)


if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    EPOCHS = 15

    for epoch in range(EPOCHS):
        for (x_train, y_train) in train_dataset:
            train_step(x_train, y_train)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            # for var in model.trainable_variables:
            #     tf.summary.histogram(var.name, var, step=epoch)
            inputs = x_train
            for var in model.layers:
                output = var(inputs)
                tf.summary.histogram(var.name, var(inputs), step=epoch)
                inputs = output

        for (x_test, y_test) in test_dataset:
            test_step(x_test, y_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            # for var in model.trainable_variables:
            #     tf.summary.histogram(var.name, var, step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()