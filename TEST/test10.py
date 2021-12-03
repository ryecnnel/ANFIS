# DeepSIRMsのトレーニング部分
# 二層です

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from RuleSIRMs2 import DeepSIRMs

import pandas as pd
df = pd.read_csv(filepath_or_buffer="./Dataset/diabetes.csv", encoding="utf-8", sep=",")

output = df.pop('output').astype(np.float64)

mc = MinMaxScaler()
mc.fit(df)

df_mc = pd.DataFrame(mc.transform(df), columns=df.columns)

dataset = tf.data.Dataset.from_tensor_slices((df_mc.values, output.values))

test = DeepSIRMs(5,3)

for i, j in dataset:
    print(test.second_layer(i))
    print(j)