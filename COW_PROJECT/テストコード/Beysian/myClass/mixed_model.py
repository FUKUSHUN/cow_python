import numpy as np
from scipy import stats # カテゴリ分布から乱数を生成

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

    def gibbs_sample(self, X, a_0, b_0):
        """ ギブスサンプリングにより各クラスタの真値を推定する
            Parameter
                X       : 1*N行列   1次元だが行列として扱う
                a_0, b_0    : ガンマ分布のハイパーパラメータ
            Return
                S   : K*N行列   予測した各クラスの割り当て
                P   : K*N行列   各データの予測確率 """
        N, K = len(X.T), len(self.lambda_vector) # データ数とクラスタ数
        S = np.zeros((K, N)) # K*N行列，1ofK表現    クラスタの割り当ての行列
        eta = np.zeros((K, N)) # K*N行列    カテゴリ分布のパラメータ
        a, b = np.zeros(K), np.zeros(K) # K次元ベクトル ガンマ分布のパラメータ
        for _ in range(self._maxiter):
            self._sample_s(X, S, eta, N, K)
            self._sample_lambda(X, S, N, K, a, b, a_0, b_0)
            self._sample_pi(S, N, K)
        return S

    def _sample_s(self, X, S, eta, N, K):
        """ S[n]をサンプルする
            Parameter
                X   : 1*N行列   入力データ. 1次元だが行列として扱う. 
                S   : K*N行列   予測した各クラスの割り当て
                N   : データ数
                K   : クラスタ数 """
        xk = np.arange(K)
        for n in range(N):
            for k in range(K):
                eta[k,n] = np.exp(X[0,n] * np.log(self.lambda_vector[k]) - self.lambda_vector[k] + np.log(self.pi_vector[k]))
            for k in range(K):
                eta[k,n] = eta[k,n] / sum(eta[k,:])          
            # todo snをサンプル
            custm = stats.rv_discrete(name='custm', values=(xk, eta[n]))
            rnd = custm.rvs(size=1)
            S[n] = np.array([1 if (k == rnd[0]) else 0 for k in range(K)])
        return S

    def _sample_lambda(self, X, S, N, K, a, b, a_0, b_0):
        """ lambda[n]をサンプルする
            Parameter
                X       : 1*N行列   入力データ. 1次元だが行列として扱う. 
                S       : K*N行列   予測した各クラスの割り当て
                N       : データ数
                K       : クラスタ数
                a, b    : K次元ベクトル ガンマ分布のパラメータ """
        sum1 = 0
        sum2 = 0
        for k in range(K):
            for n in range(N):
                sum1 += S[k,n] * X[0,n]
                sum2 += S[k,n]
            a[k] = sum1 + a_0
            b[k] = sum2 + b_0
            # todo lambda_kをサンプル
            self.lambda_vector[k] = np.random.gamma(a[k], scale=1/b[k])
        return

    def _sample_pi(self, S, N, K):
        """ piをサンプルする
            Parameter
                S   : K*N行列   予測した各クラスの割り当て
                N   : データ数
                K   : クラスタ数 """
        for k in range(K):
            for n in range(N):
                self._alpha_vector[k] += S[k,n]
        # todo pi_vectorをサンプル
        self.pi_vector = np.random.dirichlet(self._alpha_vector)
        return

    def get_lambda_vector(self):
        return self.lambda_vector
    
    def get_pi_vector(self):
        return self.pi_vector