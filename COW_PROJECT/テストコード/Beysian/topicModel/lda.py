import numpy as np
from scipy.special import psi
import pdb # デバッグ用

class GaussianLDA:
    """ Gaussian LDAによるトピック分類を行うクラス 
        D: Dimensionality of feature vector
        K: The number of topic
        N: The length of one-document's sequence
        M: The total number of documents in corpus
            by Latent Topic Model Based on Gaussian-LDA for Audio Retrieval """
    corpus: list # ドキュメント集合
    K: int # クラスタ数
    D: int # 特徴空間の次元数
    M: int # corpusにあるドキュメントの数
    _alpha: np.array # M個のドキュメント × K次元ベクトルの行列．ディリクレ分布のパラメータ (最初はすべてのドキュメントに同一のパラメータを与える)
    _W: np.array # K次元ベクトル × D×D行列の行列ウィシャート分布のパラメータ (最初はすべてのクラスタに同一のパラメータを与える)
    _nu: np.array # K次元ベクトル. ウィシャート分布の自由度パラメータ (> D-1. 最初はすべてのクラスタに同一のパラメータを与える)
    _m: np.array # K次元ベクトル × D次元ベクトルの行列. ガウス・ウィシャート分布のパラメータ
    _beta: np.array # K次元ベクトル. ガウス・ウィシャート分布のパラメータ (> 0. 最初はすべてのクラスタに同一のパラメータを与える)

    def __init__(self, corpus, num_topic, dimensionality):
        """ ドキュメント集合とトピック数を決定し，変数の確保をする """
        self.corpus = corpus # 登録
        self.K = num_topic # 登録
        self.D = dimensionality # 登録
        self.M = len(corpus) # 登録
        self._alpha = np.zeros((self.M, self.K)) # 初期化
        self._W = np.zeros((self.K, self.D, self.D)) # 初期化
        self._nu = np.zeros(self.K) # 初期化
        self._m = np.zeros((self.K, self.D)) # 初期化
        self._beta = np.zeros(self.K) # 初期化
        return

    def inference(self, alpha, psi, nu, m, beta, maxiter):
        """ 推論を行う (モデルのfit)
            alpha: np.array # K次元ベクトル．ディリクレ分布のパラメータ (最初はすべてのドキュメントに同一のパラメータを与える)
            psi: np.array # D×D行列の行列ウィシャート分布のパラメータの初期値 (最初はすべてのクラスタに同一のパラメータを与える)
            nu: int # ウィシャート分布の自由度パラメータの初期値 (> D-1. 最初はすべてのクラスタに同一のパラメータを与える)
            m: np.array # ガウス・ウィシャート分布のハイパーパラメータ (最初はすべてのクラスタに同一のパラメータを与える)
            beta: float # ガウス・ウィシャート分布のハイパーパラメータ (最初はすべてのクラスタに同一のパラメータを与える)
            maxiter: int # 反復回数 """
        self._initialize_params(alpha, psi, nu, m, beta)
        for i in range(maxiter):
            print(i, "回目の推論")
            for m, d in enumerate(self.corpus):
                # --- E step ---
                r = self._do_e_step(d, m)
                # --- M step ---
                N, x_bar, S = self._do_m_step(d, r)
                # --- inference parameters ---
                self._update_params(N, x_bar, S, m)
        return

    def _initialize_params(self, alpha, psi, nu, mm, beta):
        # initialization
        for m in range(self.M):
            self._alpha[m] = alpha
        for k in range(self.K):
            self._W[k] = psi
            self._nu[k] = nu
            self._m[k] = mm
            self._beta[k] = beta
        # sampling and first inference
        # init_theta = np.random.dirichlet(alpha) # dirichlet
        for m, d in enumerate(self.corpus):
            r = np.array([np.random.dirichlet(alpha) for _ in range(len(d))])
            # init_Z = np.random.choice([i for i in range(self.K)],p=init_theta) # Categorical
            # one_hot_Z = np.eye(self.K)[init_Z] # change into a one-hot vector
            # r = np.tile(init_theta, (len(d),1)) # prepare Z for each word in document d
            N, x_bar, S = self._do_m_step(d, r)
            self._update_params(N, x_bar, S, m)
        return

    def _do_e_step(self, doc, m):
        """ Implements of the caluculation in E step """
        r = np.zeros((len(doc), self.K)) # responsibility
        # r[n,k] を求める
        for n in range(len(doc)):
            x = doc[n]
            theta = np.zeros(self.K) # <θk>
            lam = np.zeros(self.K) # precision matrix
            for k in range(self.K):
                theta[k] = np.exp(psi(self._alpha[m,k]) - psi(sum(self._alpha[m,:])))
                tmp = 0
                for i in range(self.D):
                    tmp += psi((self._nu[k] + 1 - i)/2)
                lam[k] = (2 ** self.D) * np.linalg.det(self._W[k]) * tmp
                pdb.set_trace()
                r[n,k] = theta[k] * (lam[k] ** (1/2)) * np.exp(-1 * (self.D / (2 * self._beta[k])) -\
                     (self._nu[k] / 2) * np.dot((x - self._m[k]).T, np.dot(self._W[k], (x - self._m[k]))))
        # rを正規化する
        for n in range(len(doc)):
            sum_rn = sum(r[n,:])
            for k in range(self.K):
                r[n,k] /= sum_rn
        return r

    def _do_m_step(self, doc, r):
        """ Implements of the caluculation in M step """
        N = np.zeros(self.K) # <Zm,n>
        x_bar = np.zeros((self.K, self.D))
        S = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            for n in range(len(doc)):
                x = doc[n]
                N[k] += r[n,k]
                x_bar[k] += r[n,k] * x
            x_bar[k] /= N[k]
            for n in range(len(doc)):
                x = doc[n]
                S[k] = r[n,k] * np.dot((x - x_bar[k]), (x - x_bar[k]).T)
            S[k] /= N[k]
        return N, x_bar, S

    def _update_params(self, N, x_bar, S, m):
        """ update parameters """
        for k in range(self.K):
            old_mk = self._m[k]
            old_betak = self._beta[k]
            self._alpha[m, k] += N[k]
            self._beta[k] += N[k]
            self._nu[k] += N[k]
            self._m[k] = (1 / self._beta[k]) * (old_betak * self._m[k] + N[k] * x_bar[k])
            W_inv = np.linalg.inv(self._W[k]) + N[k] * S[k] + \
                ((old_betak * N[k]) / (old_betak + N[k])) * np.dot((x_bar[k] - old_mk), (x_bar[k] - old_mk).T)
            self._W[k] = np.linalg.inv(W_inv)
        return