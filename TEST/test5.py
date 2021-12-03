
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
        self.a1 = a.reshape(self.n+1, self.p)[:self.n,:]
        self.b1 = b.reshape(self.n+1, self.p)[:self.n,:]
        self.a2 = a.reshape(self.n+1, self.p)
        self.b2 = b.reshape(self.n+1, self.p)

        self.y1 = np.random.normal(0.5, 0.3, (self.n, self.p))
        self.w1 = np.random.normal(0.5, 0.3, (self.n))
        self.y2 = np.random.normal(0.5, 0.3, (self.n+1, self.p))
        self.w2 = np.random.normal(0.5, 0.3, (self.n+1))

    """
    @tf.function
    def h1(self, X):
        h1 = np.zeros((self.n, self.p))
        for i in range(self.n):
            #h1[i,:] = 1 - np.abs(self.a1[i,:]-X[i]) / self.b1[i,:]
            h1[i,:] = np.exp(-(self.a1[i,:] - X[i])**2 / (2*self.b1[i,:]**2))
        return h1

    @tf.function
    def y1(self, X):
        # 一層目出力
        yl = np.dot(np.sum(self.h1(X) * self.y1, axis=1) / np.sum(self.h1(X), axis=1), self.w1)
        print(yl)
    """
    @tf.function
    def first_layer(self, X):
        ###################### forward problem ############################
        # 1層目ルール群適合度
        h1 = np.zeros((self.n, self.p))
        for i in range(self.n):
            # h1[i,:] = 1 - np.abs(self.a1[i,:]-X[i]) / self.b1[i,:]
            # ガウス型
            h1[i,:] = np.exp(-(self.a1[i,:] - X[i])**2 / (2*self.b1[i,:]**2))

        # 1層目出力
        y1 = np.dot(np.sum(h1 * self.y1, axis=1) / np.sum(h1, axis=1), self.w1)

        ###################### backward problem ############################
        with tf.GradientTape(persistent=True) as t:
            t.watch(h1)
        dy_dh = t.gradient(y1, h1)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.w1)
        dy_dw = t.gradient(y1, self.w1)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a1)
        dh_da = t.gradient(h1, self.a1)
        dy_da = dy_dh * dh_da 
        
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.b1)
        dh_db = t.gradient(h1, self.b1)
        dy_db = dy_dh * dh_db 

        return y1, dy_dw, dy_da, dy_db

    @tf.function
    def second_layer(self, X, y1):
        ###################### forward problem ############################
        # 2層目入力作成
        X_ = np.append(X, y1)

        # 2層目ルール群適合度
        h2 = np.zeros((self.n+1, self.p))
        for i in range(self.n+1):
            h2[i,:] = np.exp(-(self.a2[i,:] - X_[i])**2 / (2*self.b2[i,:]**2))

        #  2層目出力
        y2 = np.dot(np.sum(h2 * self.y2, axis=1) / np.sum(h2, axis=1), self.w2)
        ###################### backward problem ############################
        with tf.GradientTape(persistent=True) as t:
            t.watch(h2)
        dy_dh = t.gradient(y2, h2)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.w2)
        dy_dw = t.gradient(y2, self.w2)

        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a2)
        dh_da = t.gradient(h2, self.a2)
        dy_da = dy_dh * dh_da 
        
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a1)
        dh_db = t.gradient(h2, self.b2)
        dy_db = dy_dh * dh_db 

        return y2, dy_dw, dy_da, dy_db
    

    @tf.function
    def loss(self, predicted_y, target_y):
        E = tf.reduce_mean(tf.square(predicted_y - target_y))

        with tf.GradientTape(persistent=True) as t:
            t.watch(predicted_y)
        dE_dy = t.gradient(E, predicted_y)

        return E, dE_dy


X_train = csv_xor_input.values[:,:2]
t = csv_xor_input.values[:,2]

X = X_train[0]
aaa = DeepSIRMs(2,2)
y1 = aaa.first_layer(X)
y2 = aaa.second_layer(X, y1)
print(y2)
# aaa.y1(0.2)