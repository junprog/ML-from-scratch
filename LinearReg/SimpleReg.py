class Model(object):
    def __init__(self, x, y): # 初期化関数 (インスタンス化に伴い、入力として x:説明変数(List型) y:目的変数(List型))
        assert len(x) == len(y), "The dim of x & y is different"
        self.x = x
        self.y = y

        self.x_mean = self._mean(x)
        self.y_mean = self._mean(y)

        self.x_var = self._var(x)
        self.y_var = self._var(y)

        self.sq_x_mean = self._mean(self._mult(x,x))
        self.xy_mean = self._mean(self._mult(x,y))

        self.a_0 = None
        self.a_1 = None

    def _mean(self, data): # リストの平均を導出する関数
        return sum(data) / len(data)

    def _var(self, data): # リストの分散を導出する関数
        mean = self._mean(data)
        sum_sub = 0.0
        for d in data:
            sum_sub += (d - mean)**2
        return sum_sub / len(data)

    def _mult(self, x, y): # リスト同士の要素積を計算する関数
        mul = []
        for x_tmp, y_tmp in zip(x, y):
            mul.append(x_tmp*y_tmp)
        return mul

    def calc_coeff(self): # 切片と係数を求める関数 (返り値に切片と係数)
        self.a_0 = (self.sq_x_mean*self.y_mean - self.x_mean*self.xy_mean) / self.x_var
        self.a_1 = (self.xy_mean - self.x_mean*self.y_mean) / self.x_var
        return self.a_0, self.a_1

    def reg(self, x): # 最適化したモデルに対して、入力に対応した予測を出力する関数
        assert self.a_0 is not None or self.a_1 is not None, "This model is not fitted."
        return self.a_0 + self.a_1*x


import numpy as np
import matplotlib.pyplot as plt

def visualize(x, y, f): # グラフ描画関数 (ライブラリとしてmatplotlib, numpyを使用)
    plt.scatter(x, y, label="datas")
    plt.title("8/1 Highest temp, Otsu city")
    plt.xlabel("Year")
    plt.ylabel("Temperature")

    np_x = np.array(x)
    plt.plot(np_x, f.reg(np_x), "r", label="regression")

    plt.legend(loc="upper left")
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

def loss_grid(x, y, a_0, a_1):

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Loss space")

    np_x = np.array(x)
    np_y = np.array(y)

    a_0_grid = np.arange(float(a_0-5000/2), float(a_0+5000/2), 10)
    a_1_grid = np.arange(float(a_1-5/2), float(a_1+5/2), 0.01)
    a_0_grid, a_1_grid = np.meshgrid(a_0_grid, a_1_grid)

    loss = 0
    for xx, yy in zip(x, y):
        loss += (1/len(x)) * ((a_0_grid+a_1_grid*xx) - yy)**2

    surf = ax.plot_surface(a_0_grid, a_1_grid, loss, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("a_0")
    ax.set_ylabel("a_1")
    ax.set_zlabel("Loss")
    plt.show()

def test(x, y, a):
    a_grid = np.arange(float(a-5/2), float(a+5/2), 0.01)

    loss = 0
    for xx, yy in zip(x, y):
        loss += (1/len(x)) * ((a_grid + 0.06500000000004158*xx) - yy)**2

    idx = np.argmin(loss)

    plt.title("Loss with griding a_0")
    plt.plot(a_grid, loss)
    plt.scatter(a, loss[idx])
    plt.xlabel("a_0")
    plt.ylabel("Loss")

    plt.show()

def test2(x, y, a):
    a_grid = np.arange(float(a-5/2), float(a+5/2), 0.01)

    loss = 0
    for xx, yy in zip(x, y):
        loss += (1/len(x)) * ((-97.62195121965237 + a_grid*xx) - yy)**2

    idx = np.argmin(loss)

    plt.title("Loss with griding a_1")
    plt.plot(a_grid, loss)
    plt.scatter(a, loss[idx])
    plt.xlabel("a_1")
    plt.ylabel("Loss")
    plt.show()

import csv

if __name__ == "__main__":

    with open("data.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        X = []
        Y = []
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))

    pred_year = 2060

    f = Model(X, Y)
    a_0, a_1 = f.calc_coeff()

    print("Coefficient (a_0, a_1) = ({}, {})".format(a_0, a_1))
    print("{} year's prediction: ".format(pred_year), f.reg(pred_year))

    visualize(X, Y, f)

    ## 予備実験 ##
    loss_grid(X, Y, a_0, a_1)

    test(X, Y, a_0)
    test2(X, Y, a_1)