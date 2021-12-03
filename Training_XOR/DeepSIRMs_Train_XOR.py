# DeepSIRMsのトレーニング部分

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
import sys
sys.path.append(os.pardir)

from RuleSIRMs.RuleSIRMs2 import DeepSIRMs
# from RuleSIRMs.RuleSIRMs3 import DeepSIRMs


import pandas as pd
df = pd.read_csv(filepath_or_buffer="./Dataset/XOR.csv", encoding="utf-8", sep=",")

print(df.head())
print(df.dtypes)
output = df.pop('output')
dataset = tf.data.Dataset.from_tensor_slices((df.values, output.values))
for inputs, outputs in dataset:
    print ('inputs: {}, output: {}'.format(inputs, outputs))
print(dataset)


## Note: このセルを再実行すると同じモデル変数が使われます
test = DeepSIRMs(2,2)
# 結果をグラフ化のために保存
train_loss_results = []
train_accuracy_results = []
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

num_epochs = 1001

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()

    # 訓練ループ
    for X, t in dataset:
        # モデルの最適化
        loss_value, grads = test.grad(X,t)
        optimizer.apply_gradients(zip(grads, test.trainable_variables))

        # 進捗の記録
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss 現在のバッチの損失を加算
        # 予測ラベルと実際のラベルを比較
        y_pred = tf.math.round(test.second_layer(X))
        # y_pred = tf.math.round(test.third_layer(X))
        epoch_accuracy.update_state([t], [y_pred])

    # エポックの終わり
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:09d}: Loss: {:.9f}, Accuracy: {:.9%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
