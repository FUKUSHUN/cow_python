import math
import numpy as np
from scipy import stats # カテゴリ分布から乱数を生成

""" ポアソン混合モデルを仮定したギブスサンプリングによるクラスタリングを行うクラス
    入力データは1次元のものを扱う """
class PoissonMixedModel:
    lambda_vector = None # K次元ベクトル  ポアソン分布の各平均パラメータのベクトル
    pi_vector = None # K次元ベクトル  カテゴリ分布の各生起確率のベクトル
    _alpha_vector = None # K次元ベクトル   ディレクレ分布のハイパーパラメータ
    _maxiter = 100
    _a = None # K次元ベクトル   ガンマ分布のパラメータaのベクトル
    _b = None # K次元ベクトル   ガンマ分布のパラメータbのベクトル

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
        self._a, self._b = np.zeros(K), np.zeros(K) # K次元ベクトル ガンマ分布のパラメータ
        for i in range(self._maxiter):
            print(i+1, "回目のサンプリング")
            self._sample_s(X, S, eta, N, K)
            self._sample_lambda(X, S, N, K, a_0, b_0)
            self._sample_pi(S, N, K)
        print(self.lambda_vector)
        print(self.pi_vector)
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
            sum_eta_n = sum(eta[:,n])
            for k in range(K):
                eta[k,n] = eta[k,n] / sum_eta_n          
            # snをサンプル
            custm = stats.rv_discrete(name='custm', values=(xk, eta[:,n]))
            rnd = custm.rvs(size=1)
            S[:,n] = np.array([1 if (k == rnd[0]) else 0 for k in range(K)])
        return

    def _sample_lambda(self, X, S, N, K, a_0, b_0):
        """ lambda[n]をサンプルする
            Parameter
                X       : 1*N行列   入力データ. 1次元だが行列として扱う. 
                S       : K*N行列   予測した各クラスの割り当て
                N       : データ数
                K       : クラスタ数 """
        for k in range(K):
            sum1 = 0
            sum2 = 0
            for n in range(N):
                sum1 += S[k,n] * X[0,n]
                sum2 += S[k,n]
            self._a[k] = sum1 + a_0
            self._b[k] = sum2 + b_0
            # lambda_kをサンプル
            self.lambda_vector[k] = np.random.gamma(self._a[k], scale=1/self._b[k])
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
        # pi_vectorをサンプル
        self.pi_vector = np.random.dirichlet(self._alpha_vector)
        return

    def predict(self, new_X):
        """ 新規入力データの出現確率を推定 (未完成だが納得いく結果になったのでここで止めている)
            工夫点としてガンマ関数がオーバーフローするのを防ぐために，対数を取った上で相殺される計算を省き近似的な結果を得ている
            Parameter
                new_X       : 1*M行列   入力データ. 1次元だが行列として扱う.  """
        M, K = len(new_X.T), len(self._a) # 入力データ数とクラスタ数
        r, p = np.zeros(K), np.zeros(K) # 負の二項分布のパラメータ
        prob = np.zeros((K,M)) # 各クラスタからの生成確率
        for k in range(K):
            r[k] = self._a[k]
            p[k] = 1 / (self._b[k] + 1)
            print(r[k])
            print(p[k])
            for m in range(M):
                equ1 = 0.0
                equ2 = 0.0
                print(k, m, "の計算")
                for x in range(1, int(new_X[0,m]+1)):
                    equ1 += math.log(x + r[k])
                print("equ1", equ1)
                for x in range(1, int(new_X[0,m]+1)):
                    equ2 += math.log(x)
                print("equ2", equ2)
                equ3 = r[k] * math.log(1-p[k]) + new_X[0,m] * math.log(p[k])
                print("equ3", equ3)
                prob[k,m] = math.exp(equ1 - equ2 + equ3)
                print("finish all")
        return prob

    def get_lambda_vector(self):
        return self.lambda_vector
    
    def get_pi_vector(self):
        return self.pi_vector