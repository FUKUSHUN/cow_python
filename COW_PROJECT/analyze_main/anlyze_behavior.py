import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d
import pandas as pd

# 3次元散布図を作成する
def show_3d_plot(df):
    # グラフ作成
    fig = plt.figure()
    ax = mpl3d.Axes3D(fig)

    # 軸ラベルの設定
    ax.set_xlabel("Last time")
    ax.set_ylabel("Current time")
    ax.set_zlabel("distance")
    #ax.set_zlabel("interval")

    # 散布図作成
    ax.plot(df['D'], df['H'], df['I'], "o", color="#00aa00", ms=4, mew=0.5)
    #ax.plot(df['D'], df['H'], df['K'], "o", color="#00aa00", ms=4, mew=0.5)

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(filepath_or_buffer = "行動解析/feature.csv", encoding = "utf-8", sep = ",", header = 0, usecols = [3,7,8,10], names=('D', 'H', 'I', 'K'))
    show_3d_plot(df)