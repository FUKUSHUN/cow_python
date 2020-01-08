import numpy as np
import math
import matplotlib.pyplot as plt

# 自作クラス
import myClass.gaussian_bayes as myGausse
import myClass.plotting as plotting


def create_artificial_data(coefficient_vector, random_varience, interval, scope):
    """ テスト用のデータセットを作成する
        Parameter
            coefficient_vector  : K * 1行列 係数ベクトル
            random_varience     : スカラー ノイズの分散 """
    deviation = np.sqrt(random_varience)
    t = np.array([interval * i for i in range(scope[0], int(scope[1]/interval)+1)])
    noises = np.random.randn(len(t)) # 標準正規分布に従う乱数を生成
    X = []
    y = []
    for i in range(len(t)):
        x = np.zeros([len(coefficient_vector), 1])
        for j in range(len(coefficient_vector)):
            x[j,0] = math.pow(t[i], j)
        y.append([np.dot(coefficient_vector.T, x)[0,0] + noises[i] * deviation])
        X.append(x.T.flatten())
    X = np.array(X).T
    y = np.array(y)
    return t, X, y

def add_stdiviation(mean_list, cov_list):
    plusone = []
    minusone = []
    for m, c in zip(mean_list, cov_list):
        st = np.sqrt(c)
        plusone.append(m + st)
        minusone.append(m - st)
    return plusone, minusone


if __name__ == '__main__':
    # 回帰のためのデータ点を生成
    coefficient_vector = np.array([[0, 3, -4, 1]]).T
    random_varience = 0.1
    scope = (0,3)
    t, _, y = create_artificial_data(coefficient_vector, 0, 0.001, scope) # ノイズなし正解の曲線
    t1, _, y1 = create_artificial_data(coefficient_vector, random_varience, 0.03, scope) # ノイズを含んだデータ点

    # 描画
    plotter = plotting.PlotUtility()
    plotter.scatter_plot(t, y, [0 for i in range(len(t))])
    plotter.scatter_plot(t1, y1, [0 for i in range(len(t1))])

    # 事前分布のパラメータ
    likelihood = [] # 周辺尤度を格納するリスト
    mean = [np.array([[0]]).T, \
            np.array([[0, 0]]).T, \
            np.array([[0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0, 0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T, \
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T]
    cov = [np.array([[1]]), \
            np.array([[1, 0], [0, 1]]), \
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), \
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), \
            np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]), \
            np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]), \
            np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]), \
            np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]), \
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]]), \
            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])]
            
            
    pre_noise = 10
    # 推論
    for i in range(len(mean)):
        _, X, _ = create_artificial_data(mean[i], 0, 0.001, scope) # i 次元の目盛りとなるi * N行列を生成
        dist = myGausse.GaussianLenearRegression(mean[i], cov[i], pre_noise)
        likelihood.append(dist.inference(X, y))
        y2, y_cov = dist.predict(X)
        plotter.scatter_plot(t, y2, [i+1 for j in range(len(t))])

    # 周辺尤度の視覚化
    ploter_likelihood = plotting.PlotUtility()
    ploter_likelihood.line_plot([i for i in range(len(mean))], likelihood)

    plotter.show()
    ploter_likelihood.show()
