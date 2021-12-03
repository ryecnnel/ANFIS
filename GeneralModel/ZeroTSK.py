import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf

import matplotlib.pyplot as plt
import contextlib

class ZeroTSK(tf.keras.layers.Layer):
    def __init__(self, input_size, part_num):
        super(ZeroTSK, self).__init__()
        self.n = input_size
        self.p = part_num

        a_ = np.linspace(0, 1, self.p)
        b_ = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n):
            a_ = np.append(a_, np.linspace(0, 1, self.p), axis=0)
            b_ = np.append(b_, np.ones(self.p) / (self.p**1.5), axis=0)
        self.a = tf.Variable(a_.reshape(self.n+1, self.p)[:self.n,:], trainable=True)
        self.b = tf.Variable(b_.reshape(self.n+1, self.p)[:self.n,:], trainable=True)

        rules_num = pow(self.p, self.n)
        y_ = np.ones(rules_num).reshape(int(rules_num/self.p), self.p)/2
        self.y = tf.Variable(y_, trainable=True)


    def layer(self, X):
        # 1層目ルール群適合度
        x = tf.reshape(X, [self.n, 1])
        h_ = tf.math.exp(-(self.a - x)**2 / (2*self.b**2))

        h = tf.experimental.numpy.outer(h_[0], h_[1])
        if self.n > 2:
            for i in range(self.n-2):
                h = tf.experimental.numpy.outer(h, h_[2+i])

        y = tf.math.reduce_sum(tf.math.reduce_sum(h * self.y) / tf.math.reduce_sum(h))
        return y

    def loss(self, X, T):
        predicted_y = self.layer(X)
        loss_value = tf.reduce_mean(tf.square(predicted_y - T))
        return loss_value


    def grad(self, X, T):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            loss_value = self.loss(X, T)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)


def main():
    X = tf.constant([0., 0.5, 1.], dtype=np.float64)
    t = tf.constant([0.2], dtype=np.float64)
    test = ZeroTSK(3,3)
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