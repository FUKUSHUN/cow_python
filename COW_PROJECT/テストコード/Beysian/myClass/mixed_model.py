import numpy as np
""" ポアソン混合モデルを仮定したギブスサンプリングによるクラスタリングを行うクラス
    入力データは1次元のものを扱う """
class PoissonMixedModel:
    lambda_vector = None # K次元ベクトル  ポアソン分布の各平均パラメータのベクトル
    pi_vector = None # K次元ベクトル  カテゴリ分布の各生起確率のベクトル
    _alpha_vector = None # K次元ベクトル   ディレクレ分布のハイパーパラメータ
    _maxiter = 100

    def __init__(self, lambda_vector, pi_vector, alpha_vector, num):
        self.lambda_vector = lambda_vector # 初期設定
        self.pi_vector = pi_vector # 初期設定
        self._alpha_vector = alpha_vector # 初期設定
        self._maxiter = num # 反復回数設定

    def gibbs_sample(self, X, a, b):
        """ ギブスサンプリングにより各クラスタの真値を推定する
            Parameter
                X       : 1*N行列   1次元だが行列として扱う
                a, b    : ガンマ分布のハイパーパラメータ
            Return
                S   : K*N行列   予測した各クラスの割り当て
                P   : K*N行列   各データの予測確率 """
        N = len(X.T)
        K = len(self.lambda_vector)
        S = np.zeros((K, N)) # K*N行列，1ofK表現    クラスタの割り当ての行列
        for i in range(self._maxiter):
            for n in range(N):
                eta_n = np.zeros(K)
                for k in range(K):
                    eta_n[k] = np.exp(X[0,n] * np.log(self.lambda_vector[k]) - self.lambda_vector[k] + np.log(self.pi_vector[k]))
                for k in range(K):
                    eta_n[k] = eta_n[k] / sum(eta_n)
                # todo snをサンプル
            for k in range(K):
                a_k = sum(S[k,n] * X[0,n]) + a
                b_k = sum(S[k,n]) + b
                # todo lambda_kをサンプル
                for n in range(N):
                    self._alpha_vector[k] += S[k,n]
            # todo pi_vectorをサンプル
    def get_lambda_vector(self):
        return self.lambda_vector
    
    def get_pi_vector(self):
        return self.pi_vector