import numpy as np
import pandas as pd
import pdb

from dbscan import DBSCAN
from plotting import PlotUtility

def create_artificial_gaussiandata(mu, cov, num):
    """ テスト用のデータセットを作成する
        Parameter
            mu      : ガウス分布の平均パラメータ (多次元)
            cov  : ガウス分布の分散共分散行列パラメータ
            num :   : データ生成個数 """
    X = np.random.multivariate_normal(mu, cov, num) # ndarray
    return X

def make_distance_matrix(X):
    """ 距離行列を作成する """
    K = len(X)
    distance_matrix = np.zeros([K, K])
    for i in range(K):
        for j in range(K):
            dist2 = (X[i, 0] - X[j, 0]) ** 2 + (X[i, 1] - X[j, 1]) ** 2
            distance_matrix[i, j] = np.sqrt(dist2)
    return distance_matrix

def devide(X, result, num):
    dataset = []
    K = len(X)
    for k, r in zip(range(K), result):
        if (r == num):
            dataset.append(X[k])
    dataset = np.array(dataset)
    return dataset

if __name__ == '__main__':
    X1 = create_artificial_gaussiandata(np.array([1, 2]), np.array([[2, 1], [1, 2]]), 20)
    X2 = create_artificial_gaussiandata(np.array([10, 8]), np.array([[2, 1], [1, 2]]), 20)
    X = np.concatenate([X1, X2], 0) # 2つのndarrayを結合
    # データの可視化
    plotter = PlotUtility()
    plotter.scatter_plot(X1[:,0], X1[:,1], [1 for _ in range(len(X1))], size=5)
    plotter.scatter_plot(X2[:,0], X2[:,1], [2 for _ in range(len(X2))], size=5)
    plotter.show()
    # クラスタリング
    dbscan = DBSCAN(2, 3)
    dist_matrix = make_distance_matrix(X)
    cluster = dbscan.fit(dist_matrix)
    print(cluster)
    # 可視化
    plotter = PlotUtility()
    for i in range(int(min(cluster)), int(max(cluster))+1):
        c = devide(X, cluster, i)
        plotter.scatter_plot(c[:,0], c[:,1], [i for _ in range(len(c))], size=5)
    plotter.show()
