"""
このコードは各種分析を行うクラスである
"""
import numpy as np
import sklearn.decomposition as skd
import sklearn.cluster as skc

"""
3次元のデータを主成分分析し，2次元にする (次元削減)
Parameter
    x, y, z : 各次元のデータのリスト
"""
def reduce_dim_from3_to2(x, y, z):
    print("今から主成分分析を行います")
    features = np.array([x, y, z]).T
    pca = skd.PCA()
    pca.fit(features)
    transformed = pca.fit_transform(features)
    print("累積寄与率: ", pca.explained_variance_ratio_)
    print("主成分分析が終了しました")
    return transformed[:, 0], transformed[:, 1]

"""
KMeansによりクラスタリングを行う (今は3次元限定)
Parameter
x, y, z : 3次元のそれぞれの次元の特徴のリスト
"""
def k_means(x, y, z, clusters):
    print("今からK-Meansによるクラスタリングを行います")
    clust_array = np.array([x, y, z])
    clust_array = clust_array.T
    pred = skc.KMeans(n_clusters = clusters).fit_predict(clust_array)
    print("K-Meansによる" + clusters + "個のクラスタに分けました")
    return pred