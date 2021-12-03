import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

fig = plt.figure(figsize=(10,7))
# figure()はFigureインスタンスを作成する。
# Figureインスタンスは、描画全体の領域を確保する。
# 引数では以下を指定できる。

# figsize: (width, height)のタプルを渡す。単位はインチ。
# dpi: 1インチあたりのドット数。
# facecolor: 背景色
# edgecolor: 枠線の色
# plt.figure()では描画領域の確保だけなので、グラフは何も描画されない。




def f1(x1, x2):
    return (2*x1 + 4*x2**2 + 0.1)**2 / 37.21

def f2(x1, x2):
    return  (4*np.sin(np.pi*x1) + 2*np.cos(np.pi*x2)+6) / 12

def f3(x1, x2):
    return ((3*np.exp(3*x1) + 2*np.exp(-4*x2))**-0.5 - 0.076) / 2.241

def f4(x1, x2, x3, x4):
    return 0.5*f1(x1, x2) + 0.5*f3(x3, x4)

def f5(x1, x2, x3, x4):
    return f1(x1, x2)*f2(x3, x4)


x1_range = np.linspace(-1., 1., 50)
x2_range = np.linspace(-1., 1., 50)


ax1 = fig.add_subplot(221, projection="3d")
ax2 = fig.add_subplot(222, projection="3d")
ax3 = fig.add_subplot(223, projection="3d")
############
# plt.figure()にグラフを描画するためにsubplotを追加する必要がある。
# subplotの追加は、add_subplotメソッドを使用する。
# 111の意味は、1行目1列の1番目という意味で、subplot(1,1,1)でも同じである。
# subplotはAxesオブジェクトを返す。
# Axesは、グラフの描画、軸のメモリ、ラベルの設定などを請け負う。

# add_subplotは基本的に上書きとなる。以下は、どういう構成化わかりすくするために、
# わざと上書きしたもの。
#############

# x と y の範囲の設定
x1, x2 = np.meshgrid(x1_range, x2_range)

# 関数を記述
f_1 = f1(x1,x2)
f_2 = f2(x1,x2)
f_3 = f3(x1,x2)

ax1.plot_surface(x1, x2, f_1, cmap = "summer")
ax2.plot_surface(x1, x2, f_2, cmap = "summer")
ax3.plot_surface(x1, x2, f_3, cmap = "summer")
# ax.contour(x1, x2, y, colors = "gray", offset = -1)  # 底面に等高線を描画

# 自動で軸を設定する場合は記述なし

# 手動で軸を設定する場合
# ax.set_xlim(-20, 20)
# ax.set_ylim(-20, 20)
# ax.set_zlim(-1500, 1500)

ax1.set_title("f1")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("f1")

ax2.set_title("f2")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("f2")

ax3.set_title("f3")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("f3")

plt.show()

import csv

train_x = [-0.9, -0.5, 0, 0.5, 0.9]
test_x = np.linspace(-1, 1, 51)

# 教師データ
with open('Dataset/NonLinearFunctions/f1_train.csv', 'w') as f:
    writer = csv.writer(f)
    for i in train_x:
        for j in train_x:
            writer.writerow([i, j, f1(i, j)])

with open('Dataset/NonLinearFunctions/f2_train.csv', 'w') as f:
    writer = csv.writer(f)
    for i in train_x:
        for j in train_x:
            writer.writerow([i, j, f2(i, j)])

with open('Dataset/NonLinearFunctions/f3_train.csv', 'w') as f:
    writer = csv.writer(f)
    for i in train_x:
        for j in train_x:
            writer.writerow([i, j, f3(i, j)])

# テストデータ
with open('Dataset/NonLinearFunctions/f1_test.csv', 'w') as f:
    writer = csv.writer(f)
    for i in test_x:
        for j in test_x:
            writer.writerow([i, j, f1(i, j)])

with open('Dataset/NonLinearFunctions/f2_test.csv', 'w') as f:
    writer = csv.writer(f)
    for i in test_x:
        for j in test_x:
            writer.writerow([i, j, f2(i, j)])

with open('Dataset/NonLinearFunctions/f3_test.csv', 'w') as f:
    writer = csv.writer(f)
    for i in test_x:
        for j in test_x:
            writer.writerow([i, j, f3(i, j)])

train_x_1 = [-0.9, 0, 0.9]
test_x_1 = np.linspace(-1, 1, 11)

# 訓練データ
with open('Dataset/NonLinearFunctions/f4_train.csv', 'w') as f:
    writer = csv.writer(f)
    for i in train_x_1:
        for j in train_x_1:
            for k in train_x_1:
                for l in train_x_1:
                    writer.writerow([i, j, k, l, f4(i, j, k, l)])

with open('Dataset/NonLinearFunctions/f5_train.csv', 'w') as f:
    writer = csv.writer(f)
    for i in train_x_1:
        for j in train_x_1:
            for k in train_x_1:
                for l in train_x_1:
                    writer.writerow([i, j, k, l, f5(i, j, k, l)])

# テストデータ

with open('Dataset/NonLinearFunctions/f4_test.csv', 'w') as f:
    writer = csv.writer(f)
    for i in test_x_1:
        for j in test_x_1:
            for k in test_x_1:
                for l in test_x_1:
                    writer.writerow([i, j, k, l, f4(i, j, k, l)])

with open('Dataset/NonLinearFunctions/f5_test.csv', 'w') as f:
    writer = csv.writer(f)
    for i in test_x_1:
        for j in test_x_1:
            for k in test_x_1:
                for l in test_x_1:
                    writer.writerow([i, j, k, l, f5(i, j, k, l)])