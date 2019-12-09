#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_time_list(column):
    time_indexes = []
    for i, _ in enumerate(column):
        time_indexes.append(i)
    return time_indexes


def devide_using_size(column):
    colors = []
    color_list = ["black", "midnightblue", "blue", "crimson", "magenta", "hotpink", "pink"]
    for item in column:
        if (item < len(color_list)):
            colors.append(color_list[item - 1])
        else:
            colors.append(color_list[len(color_list) - 1])
    return colors


def devide_using_walk(column):
    colors = []
    color_list = ["black", "midnightblue", "blue", "crimson", "magenta", "pink"]
    for item in column:
        if (item < 3):
            colors.append(color_list[0])
        elif (item < 6):
            colors.append(color_list[1])
        elif (item < 9):
            colors.append(color_list[2])
        elif (item < 12):
            colors.append(color_list[3])
        elif (item < 15):
            colors.append(color_list[4])
        else:
            colors.append(color_list[5])
    return colors


def scatter_plot(t_list, v_list, c_list):
    plt.figure(figsize=(36, 12), dpi=50)
    t = np.array(t_list)
    y = np.array(v_list)
    c = np.array(c_list)
    plt.scatter(t, y, c = c, s = 100)
    #plt.show()


def line_plot(t_list, v_list, filename=None):
	#グラフ化のための設定
	plt.plot(t_list, v_list, c="green", linestyle="dashed", linewidth=0.3)
	plt.savefig(filename)
	#plt.show()


if __name__ == '__main__':
    reading_dir = "C:\\Users\\福元駿汰\\Documents\\発表資料\\中間経過報告会\\result\\"
    writing_dir = "C:\\Users\\福元駿汰\\Documents\\発表資料\\中間経過報告会\\result\\"
    filelist = ["20299_20181015", "20299_20181016", "20299_20181017", "20299_20181018", "20299_20181019", "20299_20181020", "20299_20181021", "20299_20181022", "20299_20181023", "20299_20181024"] 
    for filename in filelist:
        print(filename)
        reading_file = reading_dir + filename + ".csv"
        df = pd.read_csv(reading_file, sep = ",", header = None, usecols = [0,1,2,3,4,5], names=('Time', 'Size', 'Rest', 'Graze', 'Walk', 'DTW')) # csv読み込み
        times = get_time_list(df['Time'])
        dtw = df['DTW']
        # コミュニティサイズを元に色を定義
        writing_filename_com = writing_dir + "community_size\\" + filename + ".png"
        colors = devide_using_size(df['Size'])
        scatter_plot(times, dtw, colors)
        line_plot(times, dtw, filename=writing_filename_com)
        # 歩行の時間を元に色を定義
        writing_filename_walk = writing_dir + "walk_proportion\\" + filename + ".png"
        colors = devide_using_walk(df['Walk'])
        scatter_plot(times, dtw, colors)
        line_plot(times, dtw, filename=writing_filename_walk)