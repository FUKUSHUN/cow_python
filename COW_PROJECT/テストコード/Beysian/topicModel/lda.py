import csv
import numpy as np
from scipy.special import psi # ディガンマ関数
from scipy.special import logsumexp # logsumexp
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
    _dir_name = "./"
    _convergence_logfile = _dir_name + "conv_log.txt"
    _parameters_logfile = _dir_name + "params_log.csv"

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
        with open(self._convergence_logfile, mode='w') as f:
            f.write("convergence check (L2-norm)\n") # 初期化
        with open(self._parameters_logfile, mode='w') as f:
            f.write("parameters")
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
        max_N = 0
        for d in self.corpus:
            max_N = len(d) if max_N < len(d) else max_N
        r = np.zeros((self.M, max_N, self.K))
        r_before = r.copy()
        for i in range(maxiter):
            print(i+1, "回目の推論")
            alpha, beta, nu, mm, W = self._alpha.copy(), self._beta.copy(), self._nu.copy(), self._m.copy(), self._W.copy()
            W_inv = np.linalg.inv(W)
            for m, d in enumerate(self.corpus):
                # --- E step ---
                r[m] = self._do_e_step(d, m, r)
                # --- M step ---
                N, x_bar, S = self._do_m_step(d, r[m])
                # --- inference parameters ---
                alpha, beta, nu, mm, W_inv = self._update_new_params(N, x_bar, S, m, alpha, beta, nu, mm, W_inv)
            self._update_params(alpha, beta, nu, mm, W_inv)
            # --- convergence check ---
            if (i % 5 == 0):
                dis = 0
                for m in range(self.M):
                    dis += self._measure_L2_norm(r[m], r_before[m])
                self._write_conv_log(i+1, dis)
                self._write_params_log(i+1)
                r_before = r.copy()
            if (dis < 0.1):
                break
        # --- inference topic ---
        Z = np.zeros((self.M, self.K))
        for m, d in enumerate(self.corpus):
            topic = self._estimate_topic(d, m, r)
            Z[m] = topic.copy()
        self._write_params_log(maxiter)
        return Z

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
        self._update_params(alpha, beta, nu, mm, W_inv)
        return

    def _do_e_step(self, doc, m, r):
        """ Implements of the caluculation in E step """
        ln_r = np.zeros((len(doc), self.K)) # アンダーフローを防ぐためにログをとる
        theta = np.zeros(self.K) # <θk>
        lam = np.zeros(self.K) # precision matrix
        # r[n,k] を求める
        for k in range(self.K):
            theta[k] = psi(self._alpha[m,k]) - psi(sum(self._alpha[m,:])) # 後でlogをとりexpが相殺されるのでexpをカット
            tmp = 0
            for i in range(self.D):
                tmp += psi((self._nu[k] + 1 - i)/2)
            lam[k] = (2 ** self.D) * np.linalg.det(self._W[k]) * np.exp(tmp)
            for n, x in enumerate(doc):
                ln_r[n,k] = theta[k] + np.log(lam[k] ** (1/2)) + (-1 * (self.D / (2 * self._beta[k])) -\
                     (self._nu[k] / 2) * np.dot((x - self._m[k]).T, np.dot(self._W[k], (x - self._m[k]))))
        # rを正規化する
        for n in range(len(doc)):
            sum_ln_rn = logsumexp(ln_r[n,:]) # logsumexp
            for k in range(self.K):
                r[m,n,k] = np.exp(ln_r[n,k] - sum_ln_rn)
        # print(r)
        return r[m]

    def _do_m_step(self, doc, r):
        """ Implements of the caluculation in M step """
        N = np.zeros(self.K) # <Zm,n>
        x_bar = np.zeros((self.K, self.D))
        S = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            # N[k], x_bar[k] を求める
            for n, x in enumerate(doc):
                N[k] += r[n,k]
                x_bar[k] += r[n,k] * x
            if (N[k] != 0):
                x_bar[k] /= N[k]
            else:
                continue
            # S[k] を求める
            for n, x in enumerate(doc):
                diff = (x - x_bar[k]).reshape((2,1))
                S[k] += r[n,k] * np.dot(diff, diff.T)
            if (N[k] != 0):
                S[k] /= N[k]
            else:
                continue
        return N, x_bar, S

    def _update_new_params(self, N, x_bar, S, m, alpha, beta, nu, mm, W_inv):
        """ update parameters """
        for k in range(self.K):
            old_betak = beta[k].copy()
            old_mk = mm[k].copy()
            alpha[m, k] += N[k]
            beta[k] += N[k]
            nu[k] += N[k]
            mm[k] = (1 / beta[k]) * (old_betak * old_mk + N[k] * x_bar[k])
            diff = (x_bar[k] - old_mk).reshape((2,1))
            W_inv[k] = W_inv[k] + (N[k] * S[k]) + \
                (((old_betak * N[k]) / (old_betak + N[k])) * np.dot(diff, diff.T))
        return alpha, beta, nu, mm, W_inv

    def _update_params(self, alpha, beta, nu, mm, W_inv):
        """ 1回反復後に変分パラメータを更新する """
        self._alpha = alpha
        self._beta = beta
        self._nu = nu
        self._m = mm
        for k in range(self.K):
            self._W[k] = np.linalg.inv(W_inv[k])
        return

    def _measure_L2_norm(self, v1, v2):
        """ 2つのベクトルのL2ノルムを測定する """
        dist = np.sqrt(np.power(v1-v2, 2).sum())
        return dist

    def _write_conv_log(self, i, dis):
        with open(self._convergence_logfile, mode='a') as f:
            f.write(str(i) + "回目の推論: " + str(dis) + "\n")
        return

    def _write_params_log(self, i):
        with open(self._parameters_logfile, mode='a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["num of iteration", i])
            writer.writerow(["alpha", self._alpha])
            writer.writerow(["beta", self._beta])
            writer.writerow(["m", self._m])
            writer.writerow(["nu", self._nu])
            writer.writerow(["W", self._W])
            writer.writerow(["\n"])
        return

    def _estimate_topic(self, doc, m, r):
        """ トピックを推定する """
        topic = np.zeros(self.K)
        r[m] = self._do_e_step(doc, m, r)
        for k in range(self.K):
            topic[k] = sum(r[m,:,k])
        sum_topic = sum(topic)
        for k in range(self.K):
            topic[k] /= sum_topic
        return topic