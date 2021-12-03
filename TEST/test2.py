# 全自動ファジィルール作成器

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

part_num = 10
a = np.linspace(0, 1, part_num)
b = np.ones(part_num) / (part_num**1.5)

#0から100まで、0.01間隔で入ったリストXを作る！
X = np.arange(0,1,0.01)
#確率密度関数にX,平均50、標準偏差20を代入
for i in range(part_num):
    #Y = norm.pdf(X,a[i],b[i])
    Y = np.exp(-(a[i] - X)**2 / (2*b[i]**2))
    plt.plot(X,Y,color='r')
plt.show()
print(a)
