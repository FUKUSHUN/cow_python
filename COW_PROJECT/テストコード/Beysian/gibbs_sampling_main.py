import numpy as np
import math
import matplotlib.pyplot as plt

# 自作クラス
import myClass.plotting as plotting

def create_artificial_data(lam, num):
    """ テスト用のデータセットを作成する
        Parameter
            lam : ポアソン分布のλパラメータ (1次元)
            num : データ生成個数 """
    X = np.random.poisson(lam, num) # ndarray
    return X

if __name__ == '__main__':
    # 多峰性の1次元データ点を生成
    X1 = create_artificial_data(20, 1000)
    X2 = create_artificial_data(50, 750)
    X = np.hstack((X1, X2)) # 2つのndarrayを結合
    np.random.shuffle(X) # データをシャッフル
    
    # データを可視化
    plotter = plotting.PlotUtility()
    plotter.hist_plot([X1,X2], 20, color=None) # ヒストグラムを表示，正解で色分け
    plotter.show()