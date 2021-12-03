import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import matplotlib.pyplot as plt
import contextlib

a = tf.Variable([[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]])
b = tf.Variable([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
x = tf.constant([0, 0.5, 1])
y = tf.constant([0.1])
Y = tf.constant(0.1)
X = tf.reshape(x, [3, 1])
w = tf.Variable([2.,3.,4.])
h = tf.math.exp(-(a - X)**2 / (2*b**2))
print(h)
print(tf.math.reduce_sum(a, axis=1))
print(a+b*1j)
#print(tf.math.reduce_sum(a * b, axis=1))
#print(tf.math.reduce_sum(a, axis=1))
#print(tf.math.reduce_sum(a * b, axis=1) /tf.math.reduce_sum(a, axis=1))
#A = tf.math.reduce_sum(a * b, axis=1) /tf.math.reduce_sum(a, axis=1)
#print((tf.math.reduce_sum(a * b, axis=1) / tf.math.reduce_sum(a, axis=1))*w)
#print(tf.math.reduce_sum(A * w))
#print(tf.concat([x,[Y]],0))
#print(y)
#print(Y)