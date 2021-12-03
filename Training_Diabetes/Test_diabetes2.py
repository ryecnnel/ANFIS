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


from ModulerFuzzy.CVIRMsLayer import CVIRMsLayer

model = tf.keras.Sequential([CVIRMsLayer(1)])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics= ['accuracy'])

model.summary()

