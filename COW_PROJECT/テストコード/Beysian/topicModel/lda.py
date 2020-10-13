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
            print(i+1, "回目の推論")
            alpha, beta, nu, mm, W = self._alpha.copy(), self._beta.copy(), self._nu.copy(), self._m.copy(), self._W.copy()
            W_inv = np.linalg.inv(W)
            for m, d in enumerate(self.corpus):
                # --- E step ---
                r = self._do_e_step(d, m)
                # --- M step ---
                N, x_bar, S = self._do_m_step(d, r)
                # --- inference parameters ---
                alpha, beta, nu, mm, W_inv = self._update_new_params(N, x_bar, S, m, alpha, beta, nu, mm, W_inv)
            self._update_params(alpha, beta, nu, mm, W_inv)
            pdb.set_trace()
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
        # random sampling and first inference
        alpha, beta, nu, mm, W = self._alpha.copy(), self._beta.copy(), self._nu.copy(), self._m.copy(), self._W.copy()
        W_inv = np.linalg.inv(W)
        for m, d in enumerate(self.corpus):
            r = np.array([np.random.dirichlet(self._alpha[m]) for _ in range(len(d))])
            N, x_bar, S = self._do_m_step(d, r)
            alpha, beta, nu, mm, W_inv = self._update_new_params(N, x_bar, S, m, alpha, beta, nu, mm, W_inv)
        # update parameters
        self._update_params(alpha, np.array([1,1,1]), np.array([20, 20, 20]), mm, W_inv)
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

    def _update_new_params(self, N, x_bar, S, m, alpha, beta, nu, mm, W_inv):
        """ update parameters """
        for k in range(self.K):
            old_betak = beta[k]
            old_mk = mm[k]
            alpha[m, k] += N[k]
            beta[k] += N[k]
            nu[k] += N[k]
            mm[k] = (1 / beta[k]) * (old_betak * old_mk + N[k] * x_bar[k])
            W_inv[k] = W_inv[k] + N[k] * S[k] + \
                ((old_betak * N[k]) / (old_betak + N[k])) * np.dot((x_bar[k] - old_mk), (x_bar[k] - old_mk).T)
        return alpha, beta, nu, mm, W_inv

    def _update_params(self, alpha, beta, nu, mm, W_inv):
        self._alpha = alpha
        self._beta = beta
        self._nu = nu
        self._m = mm
        self._W = np.linalg.inv(W_inv)
        return