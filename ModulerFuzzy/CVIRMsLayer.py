import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf

import matplotlib.pyplot as plt


class CVIRMsLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CVIRMsLayer, self).__init__(**kwargs)

    def MakeMembershipFunction(self, part_num):
        mean = np.linspace(0, 1, part_num)
        variance = np.ones(part_num) / (part_num**1.5)

        for i in range(self.n):
            mean = np.append(mean, np.linspace(0, 1, part_num), axis=0)
            variance = np.append(variance, np.ones(part_num) / (part_num**1.5), axis=0)

        Mean = mean.reshape(self.n+1, part_num)[:self.n,:]
        Variance = variance.reshape(self.n+1, part_num)[:self.n,:]

        return Mean, Variance

    def build(self, input_shape, part_num):
        self.n = input_shape
        a, b = self.MakeMembershipFunction(part_num)

        # 行列[input_shape(入力xの次元)✖️part_num(分割数)]のメンバーシップ関数を定義(平均と分散)
        self.a = tf.Variable(a, trainable=True)
        self.b = tf.Variable(b, trainable=True)

        # 初期値1の後件部実数値行列w [分割数✖️2(実部と虚部)]
        self.w = tf.Variable(tf.complex(np.ones(self.p), np.zeros(self.p)), trainable=True)

        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)

        super(CVIRMsLayer, self).build(input_shape, part_num)



    def RuleModules(self, X):
        # ルール群1
        x = tf.reshape(X, [self.n, 1])
        M = tf.math.exp(-(self.a - x)**2 / (2*self.b**2))

        R_real = tf.reduce_prod(M[::2], axis=0)
        R_imag = tf.reduce_prod(M[1::2], axis=0)
        R = tf.complex(R_real, R_imag)

        return R

    def call(self, X):
        # sum(Ri*wi)
        Z = tf.abs(tf.reduce_mean(tf.multiply(self.RuleModules(X), self.w)))
        return Z

    def loss(self, X, T):
        predicted_y = self.call(X)
        loss_value = tf.reduce_mean(tf.square(predicted_y - T))
        return loss_value


    def grad(self, X, T):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            loss_value = self.loss(X, T)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)

    def obtained_rules(self):
        Rule1 = tf.math.real(self.RuleModules([0,0]))
        Rule2 = tf.math.real(self.RuleModules([0,1]))
        Rule3 = tf.math.real(self.RuleModules([1,0]))
        Rule4 = tf.math.real(self.RuleModules([1,1]))

def main():
    X = tf.constant([0., 0.25, 0.5, 0.75, 1.], dtype=np.float64)
    t = tf.constant([0.2], dtype=np.float64)
    #test.RuleModules(X)

    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    #loss_value, grads = test.grad(X,t)
    #print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
    #for epoch in range(10000):
    #    optimizer.apply_gradients(zip(grads, test.trainable_variables))
    #    if epoch % 1000 == 0:
    #       print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(), test.loss(X,t).numpy()))


if __name__ == "__main__":
    main()