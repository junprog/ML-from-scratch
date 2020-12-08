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