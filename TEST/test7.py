import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf

import matplotlib.pyplot as plt
import contextlib

class DeepSIRMs:
    def __init__(self, input_size, part_num):
        self.n = input_size
        self.p = part_num

        a = np.linspace(0, 1, self.p)
        b = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n): 
            a = np.append(a, np.linspace(0, 1, self.p), axis=0)
            b = np.append(b, np.ones(self.p) / (self.p**1.5), axis=0)
        self.x = tf.constant(x)
        self.a1 = tf.Variable(a.reshape(self.n+1, self.p)[:self.n,:])
        self.b1 = tf.Variable(b.reshape(self.n+1, self.p)[:self.n,:])
        #print(self.a1.value())
        self.y1 = np.random.normal(0.5, 0.3, (self.n, self.p))
        self.w1 = np.random.normal(0.5, 0.3, (self.n))
        self.y2 = np.random.normal(0.5, 0.3, (self.n+1, self.p))
        self.w2 = np.random.normal(0.5, 0.3, (self.n+1))
    """
    @tf.function
    def grade_h(self, X, a, b):
        return tf.math.exp(-(a - X)**2 / (2*b**2))
    """
    def _print(self):
        print("hello!!")
        print(self.a1.value())

    def first_layer(self, X):
        # 1層目ルール群適合度
        x = tf.reshape(X, [self.n, 1])
        h1 = tf.math.exp(-(self.a1 - x)**2 / (2*self.b1**2))
        y1 = tf.math.multiply(tf.math.reduce_sum(h1 * self.y1, axis=1) / tf.math.reduce_sum(h1, axis=1), self.w1)
        return y1

    def df(self, x):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a1)
            F = self.first_layer(x)
        df_da1 = t.gradient(F, self.a1)
        print(df_da1)


x = [1.0,4.0]
y = []
z = 3.
#func = f(x, y, z)
#print(func.df())

X = tf.constant([0., 0.5, 1.], dtype=np.float64)
t = [0.2, 0.3]
test = DeepSIRMs(3,3)
test.first_layer(X)
test.df(X)
test._print()
#print(X)


