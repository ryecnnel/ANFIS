
import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
csv_xor_input = pd.read_csv(filepath_or_buffer="./Dataset/XOR.csv", encoding="utf-8", sep=",")


class DeepSIRMs:
    def __init__(self, input_size, part_num):

        self.n = input_size
        self.p = part_num

        a = np.linspace(0, 1, self.p)
        b = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n): 
            a = np.append(a, np.linspace(0, 1, self.p), axis=0)
            b = np.append(b, np.ones(self.p) / (self.p**1.5), axis=0)

        self.a1 = tf.Variable(a.reshape(self.n+1, self.p)[:self.n,:])
        self.b1 = tf.Variable(b.reshape(self.n+1, self.p)[:self.n,:])
        self.a2 = tf.Variable(a.reshape(self.n+1, self.p))
        self.b2 = tf.Variable(b.reshape(self.n+1, self.p))

        self.y1 = tf.Variable(np.random.normal(0.5, 0.3, (self.n, self.p)))
        self.w1 = tf.Variable(np.random.normal(0.5, 0.3, (self.n)))
        self.y2 = tf.Variable(np.random.normal(0.5, 0.3, (self.n+1, self.p)))
        self.w2 = tf.Variable(np.random.normal(0.5, 0.3, (self.n+1)))


    ###################### forward problem ############################
    @tf.function
    def first_layer(self, X):
        # 1層目ルール群適合度
        h1 = np.zeros((self.n, self.p))

        for i in range(self.n):
            # h1[i,:] = 1 - np.abs(self.a1[i,:]-X[i]) / self.b1[i,:]
            h1[i,:] = tf.math.exp(-(self.a1[i,:] - X[i])**2 / (2*self.b1[i,:]**2))
        return tf.math.multiply(tf.math.reduce_sum(h1 * self.y1, axis=1) / tf.math.reduce_sum(h1, axis=1), self.w1)

    @tf.function
    def second_layer(self, X):
        # 2層目入力作成
        y1 = self.first_layer(X)
        X_ = tf.concat(X, y1)

        # 2層目ルール群適合度
        h2 = tf.zeros((self.n+1, self.p))
        for i in range(self.n+1):
            h2[i,:].assign(tf.math.exp(-(self.a2[i,:] - X_[i])**2 / (2*self.b2[i,:]**2)))

        # 2層目出力
        y2 = tf.math.multiply(tf.math.reduce_sum(h2 * self.y2, axis=1) / tf.math.reduce_sum(h2, axis=1), self.w2)
        return y2

    ######################## loss function ############################
    @tf.function
    def loss(self, X, t):
        predicted_y = self.second_layer(X)
        E = tf.reduce_mean(tf.square(predicted_y - t))
        return E


    ###################### backward problem ############################
    def first_layer_BP(self, X, t):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a1)
            E = self.loss(X, t)
        dE_da = t.gradient(E, self.a1)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.b1)
            E = self.loss(X, t)
        dE_db = t.gradient(E, self.b1)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.w1)
            E = self.loss(X, t)
        dE_dw = t.gradient(E, self.w1)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.y1)
            E = self.loss(X, t)
        dE_dy = t.gradient(E, self.y1)

        self.a1 -= dE_da
        self.b1 -= dE_db
        self.w1 -= dE_dw
        self.y1 -= dE_dy
        return dE_da



    def second_layer_BP(self, X, t):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a2)
            E = self.loss(X, t)
        dE_da = t.gradient(E, self.a2)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.b2)
            E = self.loss(X, t)
        dE_db = t.gradient(E, self.b2)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.w2)
            E = self.loss(X, t)
        dE_dw = t.gradient(E, self.w2)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.y2)
            E = self.loss(X, t)
        dE_dy = t.gradient(E, self.y2)
        
        self.a2 -= dE_da
        self.b2 -= dE_db
        self.w2 -= dE_dw
        self.y2 -= dE_dy


X_train = csv_xor_input.values[:,:2]
T = csv_xor_input.values[:,2]

X = X_train[0].tolist()
t = T[0].tolist()
test = DeepSIRMs(2,2)
dE_da1 = test.first_layer_BP(X, t)
print(dE_da1)
# aaa.y1(0.2)
