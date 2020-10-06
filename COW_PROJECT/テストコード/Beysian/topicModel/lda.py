import numpy as np
import scipy as sp
import pdb # デバッグ用

class GaussianLDA:
    """ Gaussian LDAによるトピック分類を行うクラス 
        D: Dimensionality of feature vector
        K: The number of topic
        N: The length of one-document's sequence
        M: The number of documents in corpus
        by Latent Topic Model Based on Gaussian-LDA for Audio Retrieval """
    corpus: list # ドキュメント集合
    K: int # クラスタ数
    D: int # 特徴空間の次元数
    M: int # corpusにあるドキュメントの数
    N: list # i 番目の文章の単語数
    _alpha: np.array # M個のドキュメント × K次元ベクトルの行列．ディリクレ分布のパラメータ (最初はすべてのドキュメントに同一のパラメータを与える)
    _psi: np.array # K次元ベクトル × D×D行列の行列ウィシャート分布のパラメータの初期値 (最初はすべてのクラスタに同一のパラメータを与える)
    _nu: np.array # K次元ベクトル. ウィシャート分布の自由度パラメータの初期値 (> D-1. 最初はすべてのクラスタに同一のパラメータを与える)
    
    def __init__(self, corpus, num_topic, dimensionality):
        """ ドキュメント集合とトピック数を決定し，変数の確保をする """
        self.corpus = corpus # 登録
        self.K = num_topic # 登録
        self.D = dimensionality # 登録
        self.M = len(corpus) # 登録
        self.N = [len(d) for d in corpus] # 登録
        self._alpha = np.zeros((self.M, self.K)) # 初期化
        self._psi = np.zeros((self.K, self.D, self.D)) # 初期化
        self._nu = np.zeros((self.K)) # 初期化
        return

    def inference(self, alpha, psi, nu, maxiter):
        """ 推論を行う (モデルのfit)
            alpha: np.array # K次元ベクトルの行列．ディリクレ分布のパラメータ (最初はすべてのドキュメントに同一のパラメータを与える)
            psi: np.array # D×D行列の行列ウィシャート分布のパラメータの初期値 (最初はすべてのクラスタに同一のパラメータを与える)
            nu: np.array # ウィシャート分布の自由度パラメータの初期値 (> D-1. 最初はすべてのクラスタに同一のパラメータを与える)
            maxiter: int # 反復回数 """
        self._initialize_params(alpha, psi, nu)
        for i in range(maxiter):
            print(i, "回目の推論")
            for j, d in enumerate(self.corpus):
                r = np.zeros((self.N[j],self.K))
                for n in range(self.N[j]):
                    p = np.zeros(self.K)
                    l = np.zeros(self.K)
                    for k in range(self.K):
                        print("hogehoge")
                        # p[k] = 
                        # l[k] = 
                        # r[n,k] = 
        return

    def _initialize_params(self, alpha, psi, nu):
        for m in range(self.M):
            self._alpha[m] = alpha
        for d in range(self.D):
            self._psi[d] = psi
        for k in range(self.K):
            self._nu[k] = nu