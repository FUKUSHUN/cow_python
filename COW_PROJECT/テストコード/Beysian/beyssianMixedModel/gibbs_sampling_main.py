import numpy as np
import math
import matplotlib.pyplot as plt
import pdb # デバッグ用

# 自作クラス
import plotting as plotting
import mixed_model as mixed_model

# import os, sys
# # 自作メソッド
# os.chdir('../../') # カレントディレクトリを一階層上へ
# print(os.getcwd())
# sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
# import behavior_classification.models.beysian.mixed_model as mixed_model2

""" ポアソン混合モデルをギブスサンプリングによってサンプリングしクラスタリングするテストプログラム """

def create_artificial_poissondata(lam, num):
    """ テスト用のデータセットを作成する
        Parameter
            lam : ポアソン分布のλパラメータ (1次元)
            num : データ生成個数 """
    X = np.random.poisson(lam, num) # ndarray
    return X

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

def poisson_mixed_model_test():
    """ 1次元の入力データをポアソン混合モデルを用いてクラスタリングする """
    # 多峰性の1次元データ点を生成
    X1 = create_artificial_poissondata(20, 1000)
    X2 = create_artificial_poissondata(50, 750)
    X = np.hstack((X1, X2)) # 2つのndarrayを結合
    np.random.shuffle(X) # データをシャッフル
    X = np.array([X]) # データの2次元化

    # データを可視化
    plotter = plotting.PlotUtility()
    plotter.hist_plot([X1,X2], 20, color=None) # ヒストグラムを表示，正解で色分け

    # ポアソン混合モデルのパラメータの設定
    lambda_vector = np.array([30, 40])
    pi_vector = np.array([0.5, 0.5])
    alpha_vector = np.array([1, 1])
    max_iterater = 50

    # ギブスサンプリングによるクラスタリング
    a_0, b_0 = 1, 1
    poisson_model = mixed_model.PoissonMixedModel(lambda_vector, pi_vector, alpha_vector, max_iterater)
    result = poisson_model.gibbs_sample(X, a_0, b_0)

    # 新たな入力に対する確率を推定
    new_X = np.array([np.arange(1,100)])
    prob_matrix = poisson_model.predict(new_X)

    # クラスタリング結果を可視化
    X1 = extract_data(X, result, 0)
    X2 = extract_data(X, result, 1)
    plotter2 = plotting.PlotUtility()
    plotter2.hist_plot([X1,X2], 20, color=None)

    plotter_prob = plotting.PlotUtility()
    prob1, prob2 = prob_matrix[0,:], prob_matrix[1,:]
    plotter_prob.scatter_plot(new_X, prob1, [0 for _ in range(len(new_X))])
    plotter_prob.scatter_plot(new_X, prob2, [1 for _ in range(len(new_X))])

    # 表示
    plotter.show()
    plotter2.show()
    plotter_prob.show()

def gaussian_mixed_model_test():
    # 多峰性の2次元データ点を生成
    X1 = create_artificial_gaussiandata(np.array([30, 40]), np.array([[100, 25], [25, 100]]), 1100)
    X2 = create_artificial_gaussiandata(np.array([70, 20]), np.array([[150, 75], [75, 150]]), 900)
    X = np.concatenate([X1, X2], 0) # 2つのndarrayを結合
    np.random.shuffle(X) # データをシャッフル
    X = X.T

    # データの可視化
    plotter = plotting.PlotUtility()
    plotter.scatter_plot(X1[:,0], X1[:,1], [1 for _ in range(len(X1))])
    plotter.scatter_plot(X2[:,0], X2[:,1], [2 for _ in range(len(X2))])

    # ガウス混合分布のパラメータ設定
    mu_vectors = [np.array([30, 50]), np.array([70, 50])]
    cov_matrixes = [np.array([[110, 45], [45, 110]]), np.array([[130, 55], [55, 130]])]
    pi_vector = np.array([0.6, 0.4])
    alpha_vector = np.array([1, 1])
    max_iterater = 10

    # ギブスサンプリングによるクラスタリング
    gaussian_model = mixed_model.GaussianMixedModel(cov_matrixes, mu_vectors, pi_vector, alpha_vector, max_iterater)
    result = gaussian_model.gibbs_sample(X, np.array([[50, 50]]).T, 1, 3, np.array([[1, 0], [0, 1]]))

    # 新たな入力に対する確率を推定
    new_X = np.arange(1,101, 2)
    new_Y = np.arange(1,101, 2)
    grid_X, grid_Y = np.meshgrid(new_X, new_Y)
    new_X = np.array([grid_X.ravel(), grid_Y.ravel()])
    prob_matrix = gaussian_model.predict(new_X)
    
    # クラスタリング結果を可視化
    X1 = np.array(extract_data(X, result, 0))
    X2 = np.array(extract_data(X, result, 1))
    plotter2 = plotting.PlotUtility()
    plotter2.scatter_plot(X1[:,0], X1[:,1], [1 for _ in range(len(X1))])
    plotter2.scatter_plot(X2[:,0], X2[:,1], [2 for _ in range(len(X2))])

    plotter_prob = plotting.PlotUtility3D()
    prob1, prob2 = prob_matrix[0], prob_matrix[1]
    plotter_prob.plot_surface(grid_X, grid_Y, prob1.reshape([50, 50]), c=1)
    plotter_prob.plot_surface(grid_X, grid_Y, prob2.reshape([50, 50]), c=2)

    # 表示
    plotter.show()
    plotter2.show()
    plotter_prob.show()


if __name__ == '__main__':
    #poisson_mixed_model_test()
    gaussian_mixed_model_test()