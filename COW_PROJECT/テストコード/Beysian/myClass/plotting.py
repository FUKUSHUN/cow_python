"""
このコードは行動分類用にプロット関係の関数をまとめたコードである
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_toolkits.mplot3d as mpl3d

class PlotUtility:
    fig:None
    ax:None # 今のところ1行1列の図
    color_table = ['black', 'blue','green','red','yellow','pink','orangered','orange','lime','deepskyblue','gold']

    def __init__(self, fig = None, ax = None):
        if (fig is None or ax is None):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        else:
            self.fig = fig
            self.ax = ax


    def line_plot(self, t_list, v_list):
        """ 横軸に時間，縦軸にセンサ値をプロットする
        折れ線グラフ形式
        Parameter
            t_list  : 横軸の値のリスト
            v_list  : 縦軸の値のリスト """
        #グラフ化のための設定
        #self.ax = self.fig.add_subplot(1,1,1) #4行1列の図
        #ax1の設定
        #self.ax.set_xticklabels(t_list, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
        #self.ax.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
        self.ax.plot(t_list, v_list, 'b')
        #self.ax.legend(("Velocity",), loc='upper left')


    def scatter_plot(self, x_list, y_list, c_list, size=1):
        """ 横軸に時間，縦軸にセンサ値をプロットする
        散布図形式
        Parameter
            x_list  : 横軸の値のリスト
            y_list  : 縦軸の値のリスト
            c_list  : 色分けのリスト """
        color_list = [self.color_table[c] for c in c_list]
        x = np.array(x_list)
        y = np.array(y_list)
        c = np.array(color_list)
        self.ax.scatter(x, y, c = c, s = size)


    def scatter_time_plot(self, t_list, v_list, c_list, size=1):
        """ 横軸に時間，縦軸にセンサ値をプロットする
        散布図形式
        Parameter
            t_list  : 横軸の値のリスト
            v_list  : 縦軸の値のリスト
            c_list  : 色分けのリスト """
        x_list = [i for i in range(len(t_list))]
        color_list = [self.color_table[c] for c in c_list]
        x = np.array(x_list)
        y = np.array(v_list)
        c = np.array(color_list)
        self.ax.scatter(x, y, c = c, s = size)


    def show_3d_plot(self, x, y, z, c=None):
        """ 3次元散布図を作成する
        Parameter
            x, y, z : それぞれ各次元のリスト
            c   : クラスタリング済みの場合は色分けを行う（デフォルトは一色）"""
        # グラフ作成
        self.ax = mpl3d.Axes3D(self.fig)

        # 軸ラベルの設定
        self.ax.set_xlabel("First")
        self.ax.set_ylabel("Second")
        self.ax.set_zlabel("Third")

        # 散布図作成
        if (c is None):
            self.ax.scatter(x, y, z, "o", c="#00aa00")
        else:
            for x1, y1, z1, c1 in zip(x, y, z, c):
                self.ax.scatter(x1, y1, z1, "o", c=self.color_table[c1])


    def show(self):
        plt.show()

    def save_fig(self, filename):
        plt.savefig(filename)