"""
このコードは行動分類用にプロット関係の関数をまとめたコードである
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_toolkits.mplot3d as mpl3d

"""
横軸に時間，縦軸にセンサ値をプロットする
折れ線グラフ形式
Parameter
    t_list  : 横軸の値のリスト
    v_list  : 縦軸の値のリスト
"""
def line_plot(t_list, v_list):
	#グラフ化のための設定
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1) #4行1列の図
	#ax1の設定
	ax1.set_xticklabels(t_list, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
	ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
	ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H-%M")) #日付の表示形式を決定
	ax1.plot(t_list, v_list, 'b')
	ax1.legend(("Velocity",), loc='upper left')
	#plt.savefig(t_list[0].strftime("%y-%m-%d") + "-test.png")
	plt.show()

"""
横軸に時間（他のセンサ値でもよい），縦軸にセンサ値をプロットする
散布図形式
Parameter
    t_list  : 横軸の値のリスト
    v_list  : 縦軸の値のリスト
    c_list  : 色分けのリスト
"""
def scatter_plot(t_list, v_list, c_list):
	x_list = []
	for i in range(len(t_list)):
		x_list.append(i)
	x = np.array(x_list)
	y = np.array(v_list)
	c = np.array(c_list)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x, y, c = c, s = 1)
	plt.show()

"""
3次元散布図を作成する
Parameter
    x, y, z : それぞれ各次元のリスト
"""
def show_3d_plot(x, y, z):
    # グラフ作成
    fig = plt.figure()
    ax = mpl3d.Axes3D(fig)

    # 軸ラベルの設定
    ax.set_xlabel("First")
    ax.set_ylabel("Second")
    ax.set_zlabel("Third")

    # 散布図作成
    ax.scatter(x, y, z, "o", c="#00aa00")
    plt.show()

"""
時系列で2次元の散布図を作成する
Parameter
    t   : 時間のリスト
    x, y : それぞれ各次元のリスト
"""
def time_scatter(t_list, x, y):
    t = []
    for i in range(len(t_list)):
        t.append(i)
    show_3d_plot(t, x, y)