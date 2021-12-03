import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf

import matplotlib.pyplot as plt
import contextlib

class DeepSIRMs(tf.keras.layers.Layer):
    def __init__(self, input_size, part_num):
        super(DeepSIRMs, self).__init__()
        self.n = input_size
        self.p = part_num

        a = np.linspace(0, 1, self.p)
        b = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n): 
            a = np.append(a, np.linspace(0, 1, self.p), axis=0)
            b = np.append(b, np.ones(self.p) / (self.p**1.5), axis=0)
        self.a1 = tf.Variable(a.reshape(self.n+1, self.p)[:self.n,:], trainable=True)
        self.b1 = tf.Variable(b.reshape(self.n+1, self.p)[:self.n,:], trainable=True)
        self.a2 = tf.Variable(a.reshape(self.n+1, self.p), trainable=True)
        self.b2 = tf.Variable(b.reshape(self.n+1, self.p), trainable=True)

        self.y1 = tf.Variable(np.random.normal(0.5, 0.3, (self.n, self.p)), trainable=True)
        self.y2 = tf.Variable(np.random.normal(0.5, 0.3, (self.n+1, self.p)), trainable=True)
        self.w1 = tf.Variable(np.random.normal(0.5, 0.3, (self.n)), trainable=True)
        self.w2 = tf.Variable(np.random.normal(0.5, 0.3, (self.n+1)), trainable=True)


    def first_layer(self, X):
        # 1層目ルール群適合度
        x = tf.reshape(X, [self.n, 1])
        h1 = tf.math.exp(-(self.a1 - x)**2 / (2*self.b1**2))
        y1 = tf.math.reduce_sum((tf.math.reduce_sum(h1 * self.y1, axis=1) / tf.math.reduce_sum(h1, axis=1)* self.w1))
        return y1

    def second_layer(self, X):
        # 2層目入力作成
        y1 = self.first_layer(X)
        X_ = tf.concat([X, [y1]], axis=0)
        x = tf.reshape(X_, [self.n+1, 1])
        # 2層目ルール群適合度
        h2 = tf.math.exp(-(self.a2 - x)**2 / (2*self.b2**2))
        # 2層目出力
        y2 = tf.math.reduce_sum((tf.math.reduce_sum(h2 * self.y2, axis=1) / tf.math.reduce_sum(h2, axis=1)* self.w2))
        return y2

    def loss(self, X, T):
        predicted_y = self.second_layer(X)
        loss_value = tf.reduce_mean(tf.square(predicted_y - T))
        return loss_value


    def grad(self, X, T):
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch([self.a1, self.b1, self.a2, self.b2, self.w1, self.y1, self.w2, self.y2])
            tape.watch(self.trainable_variables)
            loss_value = self.loss(X, T)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)


def main():
    X = tf.constant([0., 0.5, 1.], dtype=np.float64)
    t = tf.constant([0.2], dtype=np.float64)
    test = DeepSIRMs(3,3)
    test.grad(X,t)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    loss_value, grads = test.grad(X,t)
    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
    for epoch in range(10000):
        optimizer.apply_gradients(zip(grads, test.trainable_variables))
        if epoch % 1000 == 0:
            print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(), test.loss(X,t).numpy()))


if __name__ == "__main__":
    main()