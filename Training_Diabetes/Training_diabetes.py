# DeepSIRMsのトレーニング部分

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import sys, os
sys.path.append(os.path.abspath('/Users/yuyaarai/Master1/DeepSIC/'))

# from RuleSIRMs2 import DeepSIRMs
# from RuleSIRMs3 import DeepSIRMs
# from RuleSIC.RuleSICl2 import DeepSIC
# from RuleSIC.RuleSICl3 import DeepSIC
# from GeneralModel.ZeroTSK import ZeroTSK
from GeneralModel.SIC import SIC

import pandas as pd

from ModulerFuzzy.CVIRMs import CVIRMs
df = pd.read_csv(filepath_or_buffer="./Dataset/diabetes.csv", encoding="utf-8", sep=",")
output = df.pop('output').astype(np.float64)
mc = MinMaxScaler()
mc.fit(df)
df_mc = pd.DataFrame(mc.transform(df), columns=df.columns)
dataset = tf.data.Dataset.from_tensor_slices((df_mc.values, output.values))

"""
print(df_mc.head(5))
for inputs, outputs in dataset:
    print ('inputs: {}, output: {}'.format(inputs, outputs))
"""

## Note: このセルを再実行すると同じモデル変数が使われます
test = CVIRMs(5,3)
# 結果をグラフ化のために保存
train_loss_results = []
train_accuracy_results = []
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

num_epochs = 10001

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()

    # 訓練ループ - 4個ずつのバッチを使用
    for X, t in dataset:
        # モデルの最適化
        loss_value, grads = test.grad(X,t)
        optimizer.apply_gradients(zip(grads, test.trainable_variables))

        # 進捗の記録
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss 現在のバッチの損失を加算
        # 予測ラベルと実際のラベルを比較
        y_pred = tf.math.round(test.layer(X))
        epoch_accuracy.update_state([t], [y_pred])

    # エポックの終わり
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:09d}: Loss: {:.9f}, Accuracy: {:.9%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
        

