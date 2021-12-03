# モデルのトレーニング部分

import pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# from RuleSIRMs2 import DeepSIRMs
# from RuleSRIMs3 import DeepSIRMs
from RuleSICl2 import DeepSIC
# from RuleSICl3 import DeepSIC
# from DeepSICl2Model import DeepSICModel

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
# 簡単化のため, 欠損値を削除する
dataset = dataset.dropna()

# "Origin"の列はカテゴリー（非数値）なので、ワンホットエンコーディングを行う。
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

# データセットを訓練用セットとテスト用セットに分割する。
# テスト用データセットは、作成したモデルの最終評価に使用する。
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ラベル、すなわち目的変数を特徴量から切り離しましょう。このラベルは、モデルに予測させたい数量です
train_labels = train_dataset.pop('MPG').astype(np.float64)
test_labels = test_dataset.pop('MPG').astype(np.float64)



minmaxscaler = MinMaxScaler()
minmaxscaler.fit(train_dataset)
minmaxscaler.fit(test_dataset)

normed_train_data = pd.DataFrame(minmaxscaler.transform(train_dataset), columns=train_dataset.columns)
normed_test_data = pd.DataFrame(minmaxscaler.transform(test_dataset), columns=test_dataset.columns)

input_size = len(train_dataset.keys())

train_dataset = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
test_dataset = tf.data.Dataset.from_tensor_slices((normed_test_data.values, test_labels.values))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

## Note: このセルを再実行すると同じモデル変数が使われます

test = DeepSIC(input_size, part_num=5)
# 結果をグラフ化のために保存
train_loss_results = []
train_accuracy_results = []

num_epochs = 1001

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()

    # 訓練ループ - 4個ずつのバッチを使用
    for X, t in train_dataset:
        # モデルの最適化
        loss_value, grads = test.grad(X,t)
        optimizer.apply_gradients(zip(grads, test.trainable_variables))

        # 進捗の記録
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss 現在のバッチの損失を加算
        # 予測ラベルと実際のラベルを比較
        y_pred = tf.math.round(test.second_layer(X))
        epoch_accuracy.update_state([t], [y_pred])

    # エポックの終わり
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    if epoch % 50 == 0:
        print("Epoch {:09d}: Loss: {:.9f}, Accuracy: {:.9%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
