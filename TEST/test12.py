import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf

M = tf.constant([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
print(M[1,:])
print(tf.reduce_prod(M[1::2], axis=0))

A1 = tf.constant([1.,1.,1.])
A2 = tf.constant([1.,2.,3.])
B = tf.complex(A1,A2)
print(B)
#print(tf.multiply(B,B))
print(tf.multiply(B,2.))
print(tf.reduce_mean(B))

print(tf.math.imag(B))

print(str(B.numpy().tolist()))