# モデルのトレーニング部分
import pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

import sys, os
sys.path.append(os.path.abspath('/Users/yuyaarai/Master1/DeepSIC/'))

# from RuleSIRMs2 import DeepSIRMs
# from RuleSRIMs3 import DeepSIRMs
# from RuleSIC.RuleSICl2 import DeepSIC
# from RuleSICl3 import DeepSIC
# from DeepSICl2Model import DeepSICModel

from ModulerFuzzy.CVIRMs import CVIRMs

dataset = pd.read_csv(filepath_or_buffer="./Dataset/diabetes.csv", encoding="utf-8", sep=",")

# データセットを訓練用セットとテスト用セットに分割する。
# テスト用データセットは、作成したモデルの最終評価に使用する。
train_dataset = dataset.sample(frac=0.5,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# ラベル、すなわち目的変数を特徴量から切り離す。このラベルは、モデルに予測させたい数量です
train_labels = train_dataset.pop('output').astype(np.float64)
test_labels = test_dataset.pop('output').astype(np.float64)

print(train_dataset.head())
print(test_dataset.head())

minmaxscaler = MinMaxScaler()
minmaxscaler.fit(train_dataset)
minmaxscaler.fit(test_dataset)

normed_train_data = pd.DataFrame(minmaxscaler.transform(train_dataset), columns=train_dataset.columns)
normed_test_data = pd.DataFrame(minmaxscaler.transform(test_dataset), columns=test_dataset.columns)

print(normed_train_data.head())
print(normed_train_data.dtypes)
print(normed_test_data.head())
print(normed_test_data.dtypes)

input_size = len(train_dataset.keys())

train_dataset = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
test_dataset = tf.data.Dataset.from_tensor_slices((normed_test_data.values, test_labels.values))

# 最適化関数の選択
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
