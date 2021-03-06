import csv
import math # ガンマ関数
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
    _alpha: np.array # M個のドキュメント × K次元ベクトルの行列．ディリクレ分布のパラメータ
    _W: np.array # K次元ベクトル × D×D行列の行列ウィシャート分布のパラメータ
    _nu: np.array # K次元ベクトル. ウィシャート分布の自由度パラメータ (> D-1.)
    _m: np.array # K次元ベクトル × D次元ベクトルの行列. ガウス・ウィシャート分布のパラメータ
    _beta: np.array # K次元ベクトル. ガウス・ウィシャート分布のパラメータ (> 0.)
    _dir_name = "./synchronization/topic_model/"
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
            alpha: np.array # K次元ベクトル．ディリクレ分布のパラメータ
            psi: np.array # D×D行列の行列ウィシャート分布のパラメータの初期値
            nu: int # ウィシャート分布の自由度パラメータの初期値 (> D-1.)
            m: np.array # ガウス・ウィシャート分布のハイパーパラメータ (> 0)
            beta: float # ガウス・ウィシャート分布のハイパーパラメータ
            maxiter: int # 反復回数 """
        self._initialize_params(alpha, psi, nu, m, beta)
        max_N = 0
        for d in self.corpus:
            max_N = len(d) if max_N < len(d) else max_N
        r = np.zeros((self.M, max_N, self.K))
        r_before = r.copy()
        for i in range(maxiter):
            print(i+1, "回目の推論")
            alpha, beta, nu = self._alpha.copy(), self._beta.copy(), self._nu.copy()
            z = np.zeros(self.K)
            z_x = np.zeros((self.K, self.D))
            z_xx = np.zeros((self.K, self.D, self.D))
            for m, d in enumerate(self.corpus):
                # --- E step ---
                r[m] = self._do_e_step(d, m, max_N)
                # --- M step ---
                _z, _z_x, _z_xx = self._do_m_step(d, r[m])
                z += _z
                z_x += _z_x
                z_xx += _z_xx
                # --- inference parameters ---
                alpha = self._update_new_params(_z, m, alpha)
            self._update_params(alpha, beta, nu, z, z_x, z_xx)
            # --- convergence check ---
            if (i % 10 == 0):
                dis = 0
                for m in range(self.M):
                    dis += self._measure_L2_norm(r[m], r_before[m])
                self._write_conv_log(i+1, dis)
                self._write_params_log(i+1)
                r_before = r.copy()
            if (dis < 0.1):
                break
        # --- inference topic ---
        theta = np.zeros(self.K)
        Z = np.zeros((self.M, self.K))
        for m, d in enumerate(self.corpus):
            topic = self._estimate_topic(d, m, r)
            Z[m] = topic.copy()
            theta += self._alpha[m]
        sum_theta = sum(theta)
        theta /= sum_theta # 正規化
        self._write_params_log(maxiter)
        return Z, theta

    def _initialize_params(self, alpha, psi, nu, mm, beta):
        # initialization
        self._alpha = alpha
        self._W = psi
        self._nu = nu
        self._m = mm
        self._beta = beta
        # random sampling and first inference
        alpha, beta, nu = self._alpha.copy(), self._beta.copy(), self._nu.copy()
        z = np.zeros(self.K)
        z_x = np.zeros((self.K, self.D))
        z_xx = np.zeros((self.K, self.D, self.D))
        for m, d in enumerate(self.corpus):
            r = self._do_e_step(d, m, len(d))
            _z, _z_x, _z_xx = self._do_m_step(d, r)
            z += _z
            z_x += _z_x
            z_xx += _z_xx
            alpha = self._update_new_params(_z, m, alpha)
        # update parameters
        self._update_params(alpha, beta, nu, z, z_x, z_xx)
        return

    def _do_e_step(self, doc, m, N):
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
        r = np.zeros((N, self.K))
        for n in range(len(doc)):
            sum_ln_rn = logsumexp(ln_r[n,:]) # logsumexp
            for k in range(self.K):
                r[n,k] = np.exp(ln_r[n,k] - sum_ln_rn)
        # print(r)
        return r

    def _do_m_step(self, doc, r):
        """ Implements of the caluculation in M step """
        z = np.zeros(self.K)
        z_x = np.zeros((self.K, self.D))
        z_xx = np.zeros((self.K, self.D, self.D))
        for n, x in enumerate(doc):
            x_vec = x.reshape((2, 1))
            for k in range(self.K):
                z[k] += r[n,k]
                z_x[k] += r[n,k] * x
                z_xx[k] += r[n,k] * np.dot(x_vec, x_vec.T)
        return z, z_x, z_xx

    def _update_new_params(self, z, m, alpha):
        """ update parameters """
        for k in range(self.K):
            alpha[m,k] += z[k]
        return alpha

    def _update_params(self, alpha, beta, nu, z, z_x, z_xx):
        """ 1回反復後に変分パラメータを更新する """
        W_inv = np.zeros((self.K, self.D, self.D))
        old_m = self._m.copy()
        old_beta = self._beta.copy()
        self._alpha = alpha
        self._beta = beta + z
        self._nu = nu + z
        for k in range(self.K):
            self._m[k] = ((1 / self._beta[k]) * (z_x[k] + (old_beta[k] * old_m[k])))
            old_mk = old_m[k].reshape((2,1))
            new_mk = self._m[k].reshape((2,1))
            W_inv[k] = z_xx[k] + old_beta[k] * np.dot(old_mk, old_mk.T) - self._beta[k] * np.dot(new_mk, new_mk.T) + np.linalg.inv(self._W[k])
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
            writer.writerow(["whole_alpha", [sum(self._alpha[:,k]) for k in range(self.K)]])
            writer.writerow(["beta", [beta for beta in self._beta]])
            writer.writerow(["m", [list(m_k) for m_k in self._m]])
            writer.writerow(["nu", [nu for nu in self._nu]])
            writer.writerow(["W", [[list(w_k[0]), list(w_k[1])] for w_k in self._W]])
            writer.writerow(["\n"])
        return

    def _estimate_topic(self, doc, m, r):
        """ トピックを推定する """
        topic = np.zeros(self.K)
        r[m] = self._do_e_step(doc, m, len(r[m]))
        for k in range(self.K):
            topic[k] = sum(r[m,:,k])
        sum_topic = sum(topic)
        topic /= sum_topic # 正規化
        return topic

    def set_params(self, beta, m, nu, W):
        """ パラメータを設定する (いきなり予測するとき，事前分布を与えたいときに使用する) """
        self._beta = beta
        self._m = m
        self._nu = nu
        self._W = W
        return
    
    def predict(self, new_corpus, theta):
        """ 新しいコーパス全体に対してそれぞれの文書ごとにトピック分布を算出する """
        topic_dist_list = []
        for new_doc in new_corpus:
            topic_dist = self._predict_doc(new_doc, theta)
            topic_dist_list.append(topic_dist)
        return topic_dist_list

    def _predict_doc(self, new_doc, theta):
        """ 新しい文書に対してトピックを予測する """
        topic_dist = np.zeros(self.K)
        for new_x in new_doc:
            topic_dist += self._predict_word(new_x, theta)
        sum_topic = sum(topic_dist)
        topic_dist /= sum_topic # 正規化
        return topic_dist

    def _predict_word(self, new_x, theta):
        """ 新しい単語に対してトピックを予測する
            new_x   : 新しい文書中の新しい単語
            theta   : 各トピックの出現確率のベクトル
            Return
            topic_dist: K次元ベクトル．トピック分布 """
        mu = np.zeros((self.K, self.D))
        lam = np.zeros((self.K, self.D, self.D))
        nu_hat = np.zeros(self.K)
        for k in range(self.K):
            mu[k] = self._m[k]
            lam[k] = ((1 - self.D + self._nu[k]) * self._beta[k] / (1 + self._beta[k])) * self._W[k]
            nu_hat[k] = 1 - self.D + self._nu[k]

        topic_dist = np.zeros(self.K)
        for k in range(self.K):
            temp1 = np.exp(math.lgamma((nu_hat[k] + self.D) / 2) - math.lgamma(nu_hat[k] / 2))
            temp2 = np.linalg.det(lam[k]) ** (1 / 2) / ((np.pi * nu_hat[k]) ** (self.D / 2))
            x_vec = np.array((new_x - mu[k]).reshape((2, 1)))
            temp3 = (1 + np.dot(x_vec.T, np.dot(lam[k], x_vec)) / nu_hat[k]) ** (-(nu_hat[k] + self.D) / 2)
            # topic_dist[k] = theta[k] * (temp1 * temp2 * temp3)
            topic_dist[k] = temp1 * temp2 * temp3 # トピックの出現確率を考慮しない
        sum_topic = sum(topic_dist)
        topic_dist /= sum_topic # 正規化
        return topic_dist