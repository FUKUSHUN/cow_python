import math
import numpy as np
from scipy import stats # カテゴリ分布から乱数を生成, ウィシャート分布から乱数を生成
import pdb # デバッグ用

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
                S   : K*N行列   予測した各クラスの割り当て """
        N, K = len(X.T), len(self.lambda_vector) # データ数とクラスタ数
        S = np.zeros((K, N)) # K*N行列，1ofK表現    クラスタの割り当ての行列
        eta = np.zeros((K, N)) # K*N行列    カテゴリ分布のパラメータ
        self._a, self._b = np.zeros(K), np.zeros(K) # K次元ベクトル ガンマ分布のパラメータ
        for i in range(self._maxiter):
            print(i+1, "回目のサンプリング")
            self._sample_s(X, S, eta, N, K)
            self._sample_lambda(X, S, N, K, a_0, b_0)
            self._sample_pi(S, N, K)
        print("lambda: ", self.lambda_vector)
        print("pi: ", self.pi_vector)
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


""" ガウス混合モデルを仮定したギブスサンプリングによるクラスタリングを行うクラス
    入力データは多次元のものを扱う """
class GaussianMixedModel:
    cov_matrixes = None # D*D行列をK個持つリスト ガウス分布の分散共分散行列パラメータをクラスタ数K個並べたもの
    mu_vectors = None # 1*D行列をK個持つリスト  ガウス分布の平均パラメータのベクトルをクラスタ数K個並べたもの
    pi_vector = None # K次元ベクトル  カテゴリ分布の各生起確率のベクトル
    _m_vector = None # K次元ベクトル  ガウス・ウィシャート分布のハイパーパラメータ
    _beta = None # スカラー値   ガウス・ウィシャート分布のハイパーパラメータ
    _nd = None # スカラー値 ガウス・ウィシャート分布のハイパーパラメータ
    _W = None # D*D行列 ガウス・ウィシャート分布のハイパーパラメータ
    _alpha_vector = None # K次元ベクトル   ディレクレ分布のハイパーパラメータ
    _student_parameters = None # (mu, lam, nd) 予測分布の各パラメータ，学習後にまとめてセット
    _maxiter = 100 # 最大反復回数

    def __init__(self, cov_matrixes, mu_vectors, pi_vector, alpha_vector, num):
        self.cov_matrixes = cov_matrixes
        self.mu_vectors = mu_vectors
        self.pi_vector = pi_vector
        self._alpha_vector = alpha_vector
        self._maxiter = num
    
    def gibbs_sample(self, X, m, beta, nd, W):
        """ ギブスサンプリングにより各クラスタの真値を推定する
            Parameter
                X       : D*N行列
                m       : D次元ベクトル     ガウス・ウィシャート分布のハイパーパラメータ
                beta    : >0のスカラー値    ガウス・ウィシャート分布のハイパーパラメータ
                nd      : >D-1のスカラー値  ガウス・ウィシャート分布のハイパーパラメータ
                W       : D*D行列           ガウス・ウィシャート分布のハイパーパラメータ
            Return
                S   : K*N行列   予測した各クラスの割り当て """
        D, N, K = len(X), len(X.T), len(self.pi_vector) # 次元数とデータ数とクラスタ数
        m_list = [None] * K # K個の要素を持つリスト
        beta_list = [0.0] * K # K個の要素を持つリスト
        nd_list = [0.0] * K # K個の要素を持つリスト
        W_list = [None] * K # K個の要素を持つリスト
        S = np.zeros((K, N)) # K*N行列，1ofK表現    クラスタの割り当ての行列
        eta = np.zeros((K, N)) # K*N行列    カテゴリ分布のパラメータ
        self._initialize(m_list, beta_list, nd_list, W_list, m, beta, nd, W, K)
        for i in range(self._maxiter):
            print(i+1, "回目のサンプリング")
            self._sample_s(X, S, eta, N, K, D)
            self._sample_gaussian_parameters(X, S, m_list, beta_list, nd_list, W_list, N, K, D)
            self._sample_pi(S, N, K)
        print("mu: ", self.mu_vectors)
        print("cov: ", self.cov_matrixes)
        print("pi: ", self.pi_vector)
        self._set_predict_parameters(m_list, beta_list, W_list, nd_list, D, K)
        return S

    def _initialize(self, m_list, beta_list, nd_list, W_list, m, beta, nd, W, K):
        self._m_vector, self._beta, self._nd, self._W = m, beta, nd, W
        for k in range(K):
            beta_list[k] = beta
            m_list[k] = m
            nd_list[k] = nd
            W_list[k] = W
        return

    def _sample_s(self, X, S, eta, N, K, D):
        """ S[n]をサンプルする
            Parameter
                X   : D*N行列   入力データ
                S   : K*N行列   予測した各クラスの割り当て
                N   : データ数
                K   : クラスタ数
                D   : データの次元数 """
        xk = np.arange(K)
        mu = np.zeros((K,D,1))
        lam = np.zeros((K,D,D))
        for n in range(N):
            for k in range(K):
                mu[k] = np.array([self.mu_vectors[k]]).T
                lam[k] = np.linalg.inv(self.cov_matrixes[k])
                eta[k,n] = np.exp(-1/2 * (np.dot((X[:,n:n+1] - mu[k]).T, np.dot(lam[k], (X[:,n:n+1] - mu[k])))[0,0]) + 1/2 * np.log(np.linalg.det(lam[k])) + np.log(self.pi_vector[k]))
            sum_eta_n = sum(eta[:,n])
            for k in range(K):
                eta[k,n] = eta[k,n] / sum_eta_n
            # snをサンプル
            custm = stats.rv_discrete(name='custm', values=(xk, eta[:,n]))
            rnd = custm.rvs(size=1)
            S[:,n] = np.array([1 if (k == rnd[0]) else 0 for k in range(K)])
            # pdb.set_trace()
        return
    
    def _sample_gaussian_parameters(self, X, S, m_list, beta_list, nd_list, W_list, N, K, D):
        """ ガウス分布のパラメータΛ_k, μ_kをサンプルする
            Parameter
                X   : D*N行列   入力データ
                S   : K*N行列   予測した各クラスの割り当て
                m_list, beta_list, nd_list, W_list
                N   : データ数
                K   : クラスタ数 """
        for k in range(K):
            sum_s = 0.0
            sum_sx = np.zeros((D,1))
            sum_sxx = np.zeros((D,D))
            for n in range(N):
                sum_s += S[k, n]
                sum_sx += np.dot(S[k, n], X[:,n:n+1])
                sum_sxx += np.dot(S[k, n], np.dot(X[:,n:n+1], X[:,n:n+1].T))
            # パラメータ更新，この順番に更新するのが正しい? 
            beta_list[k] = sum_s + self._beta
            m_list[k] = (sum_sx + np.dot(self._beta, self._m_vector)) / beta_list[k]
            W_list[k] = np.linalg.inv(sum_sxx + np.dot(self._beta, np.dot(self._m_vector, self._m_vector.T)) - np.dot(beta_list[k], np.dot(m_list[k], m_list[k].T)) + np.linalg.inv(self._W))
            nd_list[k] = sum_s + self._nd
            # ガウス分布のパラメータをサンプリング，この順番に更新するのが正しい
            wish = stats.wishart(df=nd_list[k],scale=W_list[k])
            self.cov_matrixes[k] = np.linalg.inv(wish.rvs(1))
            self.mu_vectors[k] = np.random.multivariate_normal(np.squeeze(m_list[k].T), 1/beta_list[k] * self.cov_matrixes[k])
            # pdb.set_trace()
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
        # pdb.set_trace()
        return

    def _set_predict_parameters(self, m_list, beta_list, W_list, nd_list, D, K):
        """ 予測分布に使用するパラメータをタプルで登録する """
        self._student_parameters = []
        for k in range(K):
            mu = m_list[k]
            lam = (1 - D + nd_list[k]) * beta_list[k] / (1 + beta_list[k]) * W_list[k]
            nd = 1 - D + nd_list[k]
            self._student_parameters.append((mu, lam, nd))
        return

    def predict(self, new_X):
        """ 新規入力に対する確率を推定する（正規化はしていない） """
        D, M, K = len(new_X), len(new_X.T), len(self._student_parameters) # 入力データ数とクラスタ数
        prob = np.zeros((K,M)) # 各クラスタからの生成確率
        sum_alpha = sum(self._alpha_vector)
        for k in range(K):
            alpha = self._alpha_vector[k] / sum_alpha
            mu = self._student_parameters[k][0]
            lam = self._student_parameters[k][1]
            nd = self._student_parameters[k][2]
            for m in range(M):
                equ1 = 0.0
                print(k, m, "の計算")
                for d in range(1, D+1):
                    equ1 += math.log((nd + d) / 2)
                equ2 = math.log(np.linalg.det(lam)) / 2 - D * math.log(math.pi * nd) / 2
                equ3 = -1 * (nd + D) * math.log(1 + 1/nd * np.dot((new_X[:,m:m+1] - mu).T, np.dot(lam, (new_X[:,m:m+1] - mu))))
                prob[k,m] = alpha * math.exp(equ1 + equ2 + equ3)
        return prob


    def get_gaussian_parameters(self):
        return self.mu_vectors, self.cov_matrixes
    
    def get_pi_vector(self):
        return self.pi_vector
