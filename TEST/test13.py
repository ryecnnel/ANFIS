# ルール部の図示に関するテスト
import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt

class CVIRMs(tf.keras.layers.Layer):
    def __init__(self, input_size, part_num):
        super(CVIRMs, self).__init__()
        self.n = input_size
        self.p = part_num
        a_ = np.linspace(0, 1, self.p)
        b_ = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n):
            a_ = np.append(a_, np.linspace(0, 1, self.p), axis=0)
            b_ = np.append(b_, np.ones(self.p) / (self.p**1.5), axis=0)
        self.a = tf.Variable(a_.reshape(self.n+1, self.p)[:self.n,:], trainable=True)
        self.b = tf.Variable(b_.reshape(self.n+1, self.p)[:self.n,:], trainable=True)

    def PlotMembFunC(self):
        fig = plt.figure(figsize=(10, 8))
        a = list(itertools.chain.from_iterable(self.a.numpy()))
        b = list(itertools.chain.from_iterable(self.b.numpy()))

        for count, (mean, variance) in enumerate(zip(a,b)):
            if count % self.p == 0:
                fig.subplots_adjust(top = 0.8, hspace = 0.8)
            print(count)
            #plt.title("input:x{}, partition:{}".format((count+1)//self.p, (count+1)%self.p))
            a = plt.subplot(self.n, self.p, count+1)
            a.set_title("input:x{}, partition:{}".format(count//self.p+1, count%self.p+1))
            X = np.arange(-0.5,1.5,0.01)
            Y = np.exp(-(mean - X)**2 / (2*variance**2))
            a.plot(X,Y,color='r')
        plt.show()

def main():
    X = tf.constant([0., 0.25, 0.5, 0.75, 1.], dtype=np.float64)
    t = tf.constant([0.2], dtype=np.float64)
    test = CVIRMs(5,4)
    test.PlotMembFunC()

if __name__ == "__main__":
    main()