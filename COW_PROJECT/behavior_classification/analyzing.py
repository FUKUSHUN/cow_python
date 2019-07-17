"""
このコードは各種分析を行うクラスである
"""
import numpy as np
import sklearn.decomposition as skd

"""
3次元のデータを主成分分析し，2次元にする
Parameter
    x, y, z : 各次元のデータのリスト
"""
def reduce_dim_from3_to2(x, y, z):
    print("今から主成分分析を行います")
    features = np.array([x.values, y.values, z.values]).T
    pca = skd.PCA()
    pca.fit(features)
    transformed = pca.fit_transform(features)
    print("累積寄与率: ", pca.explained_variance_ratio_)
    print("主成分分析が終了しました")
    return transformed[:, 0], transformed[:, 1]