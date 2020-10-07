import numpy as np
import scipy as sp
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
        self._nu = np.zeros((self.K)) # 初期化
        self._m = np.zeros((self.K, self.D)) # 初期化
        self._beta = np.zeros((self.K)) # 初期化
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
            for j, d in enumerate(self.corpus):
                r = np.zeros((len(d), self.K))
                # --- E step ---
                # r[n,k] を求める
                for n in range(len(d)):
                    x = d[n]
                    pi = np.zeros(self.K)
                    lam = np.zeros(self.K)
                    sum_alpha = sum(self._alpha[j,:])
                    for k in range(self.K):
                        pi[k] = np.exp(sp.special.psi(self._alpha[j,k]) - sp.special.psi(sum_alpha))
                        tmp = 0
                        for i in range(len(self.D)):
                            tmp += sp.special.psi((self._nu[k] + 1 - i)/2)
                        lam[k] = (2 ** self.D) * np.linalg.det(self._W[k]) * tmp
                        r[n,k] = pi[k] * (lam[k] ** (1/2)) * np.exp(-1 * (self.D / (2 * self._beta[k])) - \
                            (self._nu[k] / 2) * np.dot((x - self._m[k]).T, np.dot(self._W[k], (x - self._m[k])))) # responsibility
                # rを正規化する
                for n in range(len(d)):
                    sum_rn = sum(r[n,:])
                    for k in range(self.K):
                        r[n,k] /= sum_rn
                # --- M step ---
                N = np.zeros(self.K)
                x_bar = np.zeros((self.K, self.D))
                S = np.zeros((self.K, self.D, self.D))
                for k in range(self.K):
                    for n in range(len(d)):
                        N[k] += r[n,k]
                        x_bar[k] += r[n,k] * x
                    x_bar[k] /= N[k]
                    for n in range(len(d)):
                        S[k] = np.dot((x - x_bar[k]), (x - x_bar[k]).T)
                    S[k] /= N[k]
                # --- inference parameters ---
                for k in range(self.K):
                    self._alpha[k] += N[k]
                    self._beta[k] += N[k]
                    self._nu[k] += N[k]
                    self._m[k] = (1 / self._beta[k]) * (self._beta[k] * self._m[k] + N[k] * x_bar[k])
                    W_inv = np.linalg.inv(self._W) + N[k] * S[k] + \
                        ((self._beta[k] * N[k]) / (self._beta[k] + N[k])) * np.dot((x_bar[k] - self._m[k]), (x_bar[k] - self._m[k]).T)
                    self._W[k] = np.linalg.inv(W_inv)
        return

    def _initialize_params(self, alpha, psi, nu, m, beta):
        for m in range(self.M):
            self._alpha[m] = alpha
        for d in range(self.D):
            self._W[d] = psi
        for k in range(self.K):
            self._nu[k] = nu
            self._m[k] = m
            self._beta[k] = beta
    
    def _do_e_step(self):
        """ Implements of caluculation in E step """


    def _do_m_step(self):
        """ Implements of caluculation in M step """


    def _update_params(self):
        """ update parameters """