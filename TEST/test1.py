# tensorflow確認コード

import numpy as np
import tensorflow as tf
import contextlib

"""
# 遭遇するかもしれないいくつかのエラーをデモするためのヘルパー関数
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}: {}'.format(error_class, e))
  except Exception as e:
    print('Got unexpected exception \n  {}: {}'.format(type(e), e))
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))


class a:
    def __init__(self):
        self.a = tf.Variable([1.0,2.0])
        self.b = tf.Variable([3.0,4.0])
    
    @tf.function
    def ppp(self):
        return self.a + self.b

    def grad(self):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a)
            w = self.ppp()
            y = tf.reduce_sum(self.a)
            z = tf.multiply(y, y)
        dw_dx = t.gradient(w, self.a)
        dz_dx = t.gradient(z, self.a)  # 108.0 (4*x^3 at x = 3)
        dy_dx = t.gradient(y, self.a) 
        print(dw_dx)
        print(dz_dx)
        print(dy_dx)
    
aaa = a()
aaa.grad() 
"""
arr = np.random.normal(0.5, 0.3, (3, 3))
print(arr)
# [[-2.40461812 -2.76365861 -1.70312821]
#  [-2.29453302 -1.53210319 -1.49669024]
#  [-1.90580765 -1.45375908 -2.44137036]]

