import numpy as np
import tensorflow as tf

A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
B = np.array([2,3,4])
X = np.outer(A,B)
C = np.array([1,2,3])
Y = np.outer(X,C)

input = 4
A_ = tf.Variable(A)
h = tf.experimental.numpy.outer(A_[0], A_[1])

if input > 2:
    for i in range(input-2):
        h = tf.experimental.numpy.outer(h, A_[2+i])
rules = pow(3,input)
#y = tf.Variable(np.ones(rules).reshape(3,rules/3))
y = np.ones(rules).reshape(int(rules/3), 3)/2
y_ = tf.Variable(y)
print(y_)