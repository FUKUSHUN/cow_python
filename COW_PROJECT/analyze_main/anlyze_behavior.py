import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3d
import pandas as pd

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
    ax.plot(a, df['J'], df['L'], "o", color="#00aa00", ms=4, mew=0.5)
    #ax.plot(df['D'], df['H'], df['K'], "o", color="#00aa00", ms=4, mew=0.5)

    plt.show()

#datetime型のデータを整数型に直して系列を作成する
def translate_dt_to_int(dt_list):
    i_list = []
    for i, dt in enumerate(dt_list):
        i_list.append(i)
    return pd.Series(i_list)

if __name__ == '__main__':
    df = pd.read_csv(filepath_or_buffer = "feature.csv", encoding = "utf-8", sep = ",", header = 0, usecols = [0,3,7,8,9,11], names=('A', 'D', 'H', 'I', 'J','L'))
    show_3d_plot(df)