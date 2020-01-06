import numpy as np

class GaussianLenearRegression:
    """ ベイズ線形回帰を行うクラス 
        y = w * x + noise   : xはD次元，yは1次元 """
    mean_vector: None # 平均パラメータ
    cov_matrix: None # 共分散行列
    noise_var: None # ノイズの分散(1次元)

    def __init__(self, pre_mean, pre_cov, pre_noise):
        """ 事前分布の設定を行う．精度ではなく分散をパラメータとして持っていることに注意
            Parameter
                pre_mean:   事前分布の平均パラメータ
                pre_cov :   事前分布の共分散行列パラメータ
                pre_noise:  yに含まれるノイズの分散 """
        self.mean_vector = pre_mean
        self.cov_matrix = pre_cov
        self.noise_var = pre_noise


    def inference(self, X, y):
        """ 学習を行う (引数がどちらも行列であることに注意numpyの2次元配列で渡すこと)
            Parameter
                X   : K * N行列 :X = [x1, x2, ..., xn].Tの形状をしているものとして扱う
                y   : 1 * N行列 :y = [y1, y2, ..., yn].Tの形状. xの出力値 """
        lam = 1 / self.noise_var # noiseの精度（分散の逆行列）
        gam = np.linalg.inv(self.cov_matrix) # wの精度（分散の逆行列）
        sum_x = np.zeros([len(X), len(X)])
        sum_xy = np.zeros([len(X), 1])
        for n in range(len(X.T)):
            sum_x += np.dot(np.array([X[:,n]]).T, np.array([X[:,n]])) # X[n]だと1次元のため転置がうまく効かないnumpy仕様
            sum_xy += y[n,0] * np.array([X[:,n]]).T # X[n]だと1次元のため転置がうまく効かないnumpy仕様
        new_gam = lam * sum_x + gam # 事後分布の分散パラメータ
        new_m = np.dot(np.linalg.inv(new_gam), (lam * sum_xy + np.dot(gam, self.mean_vector)))
        likelihood = self._evaluate(y, new_m, new_gam)
        self.cov_matrix = np.linalg.inv(new_gam)
        self.mean_vector = new_m
        return likelihood[0,0]


    def predict(self, X):
        """ 予測を行う 
            Parameter
                X   : K * N行列 :X = [x1, x2, ..., xn].Tの形状をしているものとして扱う """
        lam = 1 / self.noise_var
        y_pred = []
        y_cov = []
        for x_new in X.T:
            y_mean = np.dot(self.mean_vector.T, x_new)
            y_lambda = lam + np.dot(x_new.T, np.dot(self.cov_matrix, x_new))
            y_pred.append(y_mean)
            y_cov.append(y_lambda)
        y_pred = np.array(y_pred)
        y_cov = np.array(y_cov)
        return y_pred, y_cov


    def _evaluate(self, y, mean_new, cov_new):
        """ 周辺尤度の計算を行う """
        lam = 1 / self.noise_var
        s = 0
        for n in range(len(y)):
            s += lam * y[n, 0] * y[n, 0] + np.log(lam) + np.log(2 * np.pi)
        likelihood = -1 / 2 * (\
            s + np.dot(self.mean_vector.T, np.dot(self.cov_matrix, self.mean_vector)) \
            - np.log(np.linalg.det(self.cov_matrix)) \
            - np.dot(mean_new.T, np.dot(cov_new, mean_new)) \
            + np.log(np.linalg.det(cov_new)))
        return likelihood


    def get_mean_vector(self):
        return self.mean_vector

    def get_cov_matrix(self):
        return self.cov_matrix