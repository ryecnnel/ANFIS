import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import os

class CVIRMs(tf.keras.layers.Layer):
    def __init__(self, input_size, part_num):
        super(CVIRMs, self).__init__()
        self.n = input_size
        self.p = part_num

        # 行列[input_size(入力xの次元)✖️part_num(分割数)]のメンバーシップ関数を定義(平均と分散)
        a_ = np.linspace(0, 1, self.p)
        b_ = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n):
            a_ = np.append(a_, np.linspace(0, 1, self.p), axis=0)
            b_ = np.append(b_, np.ones(self.p) / (self.p**1.5), axis=0)
        self.a = tf.Variable(a_.reshape(self.n+1, self.p)[:self.n,:], trainable=True)
        self.b = tf.Variable(b_.reshape(self.n+1, self.p)[:self.n,:], trainable=True)

        # 初期値1の後件部実数値行列w [分割数✖️2(実部と虚部)]
        self.w = tf.Variable(tf.complex(np.ones(self.p), np.zeros(self.p)), trainable=True)


    def RuleModules(self, X):
        # ルール群1
        x = tf.reshape(X, [self.n, 1])
        M = tf.math.exp(-(self.a - x)**2 / (2*self.b**2))
        R_real = tf.reduce_prod(M[::2], axis=0)
        R_imag = tf.reduce_prod(M[1::2], axis=0)
        R = tf.complex(R_real, R_imag)
        return R


    def layer(self, X):
        Z = tf.abs(tf.reduce_mean(tf.multiply(self.RuleModules(X), self.w)))
        return Z


    def loss(self, X, T):
        predicted_y = self.layer(X)
        loss_value = tf.reduce_mean(tf.square(predicted_y - T))
        return loss_value


    def grad(self, X, T):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            loss_value = self.loss(X, T)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)


    def MakeObtainedRulesFile(self, txt_title, dataset_name):
        new_dir_path = os.path.abspath('/Users/yuyaarai/Master1/DeepSIC/ObtainedRules/'+txt_title+'')
        os.mkdir(new_dir_path)

        f = open('ObtainedRules/' + txt_title + '/detail.txt', 'w')
        f.write("Dataset: {}\n".format(dataset_name))
        f.close()


    def ObtainedRules(self, txt_title, Epochs):
        a_list = self.a.numpy().tolist()
        b_list = self.b.numpy().tolist()
        w_list = self.w.numpy().tolist()
        w_amplitude_list = tf.math.abs(self.w).numpy().tolist()
        w_phase_list = tf.math.angle(self.w).numpy().tolist()
        f = open('ObtainedRules/'+ txt_title +'/detail.txt', 'a')
        f.write("Epochs for learning: {}, Input number: {}, Fuzzy partition number: {}\n\n".format(Epochs, self.n, self.p))
        f.write("Adjusted Membership Function\n")
        f.write("a (each mean of Membership Functions)\n")
        f.write(str(a_list) + "\n\n")
        f.write("b (each variance of Membership Functions)\n")
        f.write(str(b_list) + "\n\n")
        f.write("w (Complex number: Obtained rotation and expansion)\n")
        f.write(str(w_list) + "\n\n")
        f.write("w (amplitude)\n")
        f.write(str(w_amplitude_list) + "\n\n")
        f.write("w (phase)\n")
        f.write(str(w_phase_list) + "\n\n")
        f.close()


    def PlotMembFunC(self, txt_title):
        fig = plt.figure(figsize=(10, 8))
        a = list(itertools.chain.from_iterable(self.a.numpy()))
        b = list(itertools.chain.from_iterable(self.b.numpy()))

        for count, (mean, variance) in enumerate(zip(a,b)):
            if count % self.p == 0:
                fig.subplots_adjust(top = 0.8, hspace = 0.8)
            #plt.title("input:x{}, partition:{}".format((count+1)//self.p, (count+1)%self.p))
            a = plt.subplot(self.n, self.p, count+1)
            a.set_title("input:x{}, partition:{}".format(count//self.p+1, count%self.p+1))
            X = np.arange(-0.5,1.5,0.01)
            Y = np.exp(-(mean - X)**2 / (2*variance**2))
            a.plot(X,Y,color='r')
        plt.show()
        fig.savefig("ObtainedRules/" + txt_title + "/ObtainedMembFunc.png")


def main():
    X = tf.constant([0., 0.25, 0.5, 0.75, 1.], dtype=np.float64)
    t = tf.constant([0.2], dtype=np.float64)
    test = CVIRMs(5,3)
    #test.RuleModules(X)
    #test.MakeObtainedRulesFile("testtxt", "sampledataset")
    test.ObtainedRules("testtxt", 10000)
    test.PlotMembFunC("testtxt")

    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    #loss_value, grads = test.grad(X,t)
    #print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
    #for epoch in range(10000):
    #    optimizer.apply_gradients(zip(grads, test.trainable_variables))
    #    if epoch % 1000 == 0:
    #       print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(), test.loss(X,t).numpy()))


if __name__ == "__main__":
    main()