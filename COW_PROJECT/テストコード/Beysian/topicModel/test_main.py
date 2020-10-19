import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pdb # デバッグ用

# 自作クラス
import plotting as plotting
import lda as lda

""" トピックモデルを変分推論によって近似しクラスタリングするテストプログラム """

def create_artificial_gaussiandata(mu, cov, num):
    """ テスト用のデータセットを作成する
        Parameter
            mu      : ガウス分布の平均パラメータ (多次元)
            cov  : ガウス分布の分散共分散行列パラメータ
            num :   : データ生成個数 """
    X = np.random.multivariate_normal(mu, cov, num) # ndarray
    return X

def extract_data(X, S, k):
    """ Sの結果からk番目のクラスタに所属するデータをXから抽出する """
    N = len(X.T)
    new_X = []
    for n in range(N):
        if (S[k, n] == 1):
            new_X.append(X[:,n])
    return new_X

def gaussian_mixed_model_test():
    # 多峰性の2次元データ点を生成
    X1 = create_artificial_gaussiandata(np.array([0.4, 0.3]), np.array([[0.006, 0.0025], [0.0025, 0.006]]), 1100)
    X2 = create_artificial_gaussiandata(np.array([0.1, 0.2]), np.array([[0.002, 0.001], [0.001, 0.002]]), 900)
    X3 = create_artificial_gaussiandata(np.array([0.3, 0.6]), np.array([[0.003, 0.005], [0.005, 0.003]]), 500)

    # データの可視化
    plotter = plotting.PlotMaker2D(0, 1, 0, 1)
    plotter.set_lim(True, True)
    plotter.create_scatter_plot(X1[:,0], X1[:,1], color=["red" for _ in range(len(X1))])
    plotter.create_scatter_plot(X2[:,0], X2[:,1], color=["green" for _ in range(len(X2))])
    plotter.create_scatter_plot(X3[:,0], X3[:,1], color=["blue" for _ in range(len(X3))])

    # 文書集合の作成
    corpus1 = []
    for i in range(800):
        numof_words = min(10, 1 + int(np.random.rand() * i)) # 単語数
        doc = np.zeros((numof_words, 2))
        np.random.shuffle(X1)
        for j in range(numof_words):
            doc[j] = X1[j]
        corpus1.append(doc)
    corpus2 = []
    for i in range(500):
        numof_words = min(10, 1 + int(np.random.rand() * i)) # 単語数
        doc = np.zeros((numof_words, 2))
        np.random.shuffle(X2)
        for j in range(numof_words):
            doc[j] = X2[j]
        corpus2.append(doc)
    corpus3 = []
    for i in range(300):
        numof_words = min(10, 1 + int(np.random.rand() * i)) # 単語数
        doc = np.zeros((numof_words, 2))
        np.random.shuffle(X3)
        for j in range(numof_words):
            doc[j] = X3[j]
        corpus3.append(doc)
    # plotter = plotting.PlotMaker2D(0, 1, 0, 1)
    # plotter.set_lim(True, True)
    # for doc in corpus1:
    #     plotter.create_scatter_plot(doc[:,0], doc[:,1], color=["red" for _ in range(len(doc))])
    # for doc in corpus2:
    #     plotter.create_scatter_plot(doc[:,0], doc[:,1], color=["green" for _ in range(len(doc))])
    # for doc in corpus3:
    #     plotter.create_scatter_plot(doc[:,0], doc[:,1], color=["blue" for _ in range(len(doc))])
    # plotter.show()


    # ガウス混合分布のパラメータ設定
    alpha = np.array([1, 1, 1]) # parameter for dirichlet
    psi = np.array([[1, 0], [0, 1]]) # parameter for Gaussian Wishert
    m = np.array([0.4, 0.4]) # parameter for Gaussian Wishert
    nu = 1 # parameter for Gaussian Wishert
    beta = 1 # parameter for Gaussian Wishert
    max_iter = 2000

    # ギブスサンプリングによるクラスタリング
    corpus = corpus1 + corpus2 + corpus3
    gaussian_lda = lda.GaussianLDA(corpus = random.sample(corpus, len(corpus)), num_topic=3, dimensionality=2)
    Z = gaussian_lda.inference(alpha, psi, nu, m, beta, max_iter)
    pdb.set_trace()

    # # 新たな入力に対する確率を推定
    # new_X = np.arange(1,101, 2)
    # new_Y = np.arange(1,101, 2)
    # grid_X, grid_Y = np.meshgrid(new_X, new_Y)
    # new_X = np.array([grid_X.ravel(), grid_Y.ravel()])
    # prob_matrix = gaussian_model.predict(new_X)
    
    # # クラスタリング結果を可視化
    # X1 = np.array(extract_data(X, result, 0))
    # X2 = np.array(extract_data(X, result, 1))
    # plotter2 = plotting.PlotUtility()
    # plotter2.scatter_plot(X1[:,0], X1[:,1], [1 for _ in range(len(X1))])
    # plotter2.scatter_plot(X2[:,0], X2[:,1], [2 for _ in range(len(X2))])

    # plotter_prob = plotting.PlotUtility3D()
    # prob1, prob2 = prob_matrix[0], prob_matrix[1]
    # plotter_prob.plot_surface(grid_X, grid_Y, prob1.reshape([50, 50]), c=1)
    # plotter_prob.plot_surface(grid_X, grid_Y, prob2.reshape([50, 50]), c=2)

    # # 表示
    # plotter.show()
    # plotter2.show()
    # plotter_prob.show()


if __name__ == '__main__':
    gaussian_mixed_model_test()