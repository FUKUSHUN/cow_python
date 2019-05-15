import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d
import pandas as pd
import numpy as np
import sklearn.decomposition as skd

# 3次元散布図を作成する
def show_3d_plot(df):
    # グラフ作成
    fig = plt.figure()
    ax = mpl3d.Axes3D(fig)

    # 軸ラベルの設定
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance")
    ax.set_zlabel("Interval")
    #ax.set_zlabel("interval")

    a = translate_dt_to_int(df['A'])
    # 散布図作成
    #A:Last end time, B:Last latitude, C:Last longitude, D:last continuous time, E:Start time, F:Latitude, G:Longitude, H:Continuous time, I:Moving distance, J:Moving amount, K:Moving direction, L:Interval between rests
    ax.plot(df['H'], df['J'], df['L'], "o", color="#00aa00", ms=4, mew=0.5)
    #plt.show()
    #ax.plot(df['D'], df['H'], df['K'], "o", color="#00aa00", ms=4, mew=0.5)
    b, c = reduce_dim_from3_to2(df['H'], df['J'], df['L'])
    ax.plot(a, b, c, "o", color="#00aa00", ms=4, mew=0.5)
    plt.show()

#datetime型のデータを整数型に直して系列を作成する
def translate_dt_to_int(dt_list):
    i_list = []
    for i, dt in enumerate(dt_list):
        i_list.append(i)
    return pd.Series(i_list)

#3次元のデータを主成分分析し，2次元にする
def reduce_dim_from3_to2(x, y, z):
    print("今から主成分分析を行います")
    features = np.array([x.values, y.values, z.values]).T
    pca = skd.PCA()
    pca.fit(features)
    transformed = pca.fit_transform(features)
    print("累積寄与率: ", pca.explained_variance_ratio_)
    print("今から主成分分析を行いました")
    return transformed[:, 0], transformed[:, 1]

if __name__ == '__main__':
    #0:Last end time, 1:Last latitude, 2:Last longitude, 3:last continuous time, 4:Start time, 5:Latitude, 6:Longitude, 7:Continuous time, 8:Moving distance, 9:Moving amount, 10:Moving direction, 11:Interval between rests
    df = pd.read_csv(filepath_or_buffer = "feature.csv", encoding = "utf-8", sep = ",", header = 0, usecols = [0,3,7,8,9,11], names=('A', 'D', 'H', 'I', 'J','L'))
    show_3d_plot(df)