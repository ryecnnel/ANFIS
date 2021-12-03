# DeepSIRMsのトレーニング部分

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from RuleSIRMs.RuleSIRMs1 import DeepSIRMs
# from RuleSIRMs.RuleSIRMs2 import DeepSIRMs
# from RuleSIRMs.RuleSIRMs3 import DeepSIRMs


import pandas as pd
df_train = pd.read_csv(filepath_or_buffer="./Dataset/NonLinearFunctions/f5_train.csv", encoding="utf-8", sep=",")
df_test = pd.read_csv(filepath_or_buffer="./Dataset/NonLinearFunctions/f5_test.csv", encoding="utf-8", sep=",")


print(df_train.head())
print(df_train.dtypes)
print(df_test.head())
print(df_test.dtypes)
output_train = df_train.pop('output')
output_test = df_test.pop('output')
dataset = tf.data.Dataset.from_tensor_slices((df_train.values, output_train.values))
dataset_for_test = tf.data.Dataset.from_tensor_slices((df_test.values, output_test.values)) 


## Note: このセルを再実行すると同じモデル変数が使われます
test = DeepSIRMs(4,5)
# 結果をグラフ化のために保存
train_loss_results = []
train_accuracy_results = []
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

num_epochs = 10001

for epoch in range(num_epochs):
    epoch_train_loss_avg = tf.keras.metrics.Mean()
    epoch_test_loss_avg = tf.keras.metrics.Mean()

    # 訓練ループ - 4個ずつのバッチを使用
    for X, t in dataset:
        # モデルの最適化
        loss_value, grads = test.grad(X,t)
        optimizer.apply_gradients(zip(grads, test.trainable_variables))

        # 進捗の記録
        epoch_train_loss_avg.update_state(loss_value)  # Add current batch loss 現在のバッチの損失を加算


    if epoch % 500 == 0:
        for X, t in dataset_for_test:
            # モデルの最適化
            loss_value = test.loss(X,t)

            # 進捗の記録
            epoch_test_loss_avg.update_state(loss_value)  # Add current batch loss 現在のバッチの損失を加算

            # エポックの終わり
            train_loss_results.append(epoch_test_loss_avg.result())

        print("Epoch {:09d}; Train Loss: {:.9f};  Test Loss: {:.9f}".format(epoch, epoch_train_loss_avg.result(), epoch_test_loss_avg.result()))
