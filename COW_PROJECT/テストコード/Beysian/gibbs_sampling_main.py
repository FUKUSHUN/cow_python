import numpy as np
import math
import matplotlib.pyplot as plt

# 自作クラス
import myClass.plotting as plotting
import myClass.mixed_model as mixed_model

def create_artificial_data(lam, num):
    """ テスト用のデータセットを作成する
        Parameter
            lam : ポアソン分布のλパラメータ (1次元)
            num : データ生成個数 """
    X = np.random.poisson(lam, num) # ndarray
    return X

def extract_data(X, S, k):
    N = len(X.T)
    new_X = []
    for n in range(N):
        if (S[k, n] == 1):
            new_X.append(X[0,n])
    return new_X

if __name__ == '__main__':
    # 多峰性の1次元データ点を生成
    X1 = create_artificial_data(20, 1000)
    X2 = create_artificial_data(50, 750)
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
