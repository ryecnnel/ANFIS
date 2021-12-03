# グレード計算機 第2層まで

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import tensorflow as tf

input_size = 4
part_num = 3

a = np.linspace(0, 1, part_num)
for i in range(input_size-1): a = np.append(a, np.linspace(0, 1, part_num), axis=0)
a = a.reshape(input_size, part_num)[:input_size-1,:]
#print(a)

class DeepSIRMs:
    l = 2 # 層の数
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
    def first_layer(self, X):
        # 1層目ルール群適合度
        h1 = np.zeros((self.n, self.p))
        for i in range(self.n):
            # h1[i,:] = 1 - np.abs(self.a1[i,:]-X[i]) / self.b1[i,:]
            # ガウス型
            h1[i,:] = np.exp(-(self.a1[i,:] - X[i])**2 / (2*self.b1[i,:]**2))

        # 1層目出力
        y1 = np.dot(np.sum(h1 * self.y1, axis=1) / np.sum(h1, axis=1), self.w1)
        return h1, y1

    def second_layer(self, X, y1):
        # 2層目入力作成
        X_ = np.append(X, y1)

        # 2層目ルール群適合度
        h2 = np.zeros((self.n+1, self.p))
        for i in range(self.n+1):
            h2[i,:] = np.exp(-(self.a2[i,:] - X_[i])**2 / (2*self.b2[i,:]**2))

        # 1層目出力
        y2 = np.dot(np.sum(h2 * self.y2, axis=1) / np.sum(h2, axis=1), self.w2)
        
        return y2


X = [0.2,0.5]
aaa = DeepSIRMs(2,3)
_,y1 = aaa.first_layer(X) 
y2 = aaa.second_layer(X, y1)
print(y2)
# aaa.y1(0.2)