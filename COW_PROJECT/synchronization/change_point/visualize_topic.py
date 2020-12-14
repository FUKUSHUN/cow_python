"""
このコードは行動分類用にプロット関係の関数をまとめたコードである
"""
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_toolkits.mplot3d as mpl3d
import pdb

def plot_by_day(df, usecols, names, filename):
    color_list = ['red', 'green', 'blue', 'orange', 'purple']
    fig = plt.figure(figsize=(19.2, 9.67))
    plt.subplots_adjust(left=0.10, right=0.90, bottom=0.05, top=0.95, hspace=0.35)
    for i in range(len(usecols)):
        ax = fig.add_subplot(len(usecols),1,i+1)
        if (i < len(usecols) - 1):
            _plot_heatmap(ax, df[names[i]], names[i], color_list[i])
        else:
            _plot_color_bar(ax, df[names[i]], names[i], color_list)
    plt.savefig(filename)
    plt.close()
    print(filename + "を作成しました")
    return

def plot_exception(names, filename):
    fig = plt.figure(figsize=(19.2, 9.67))
    plt.subplots_adjust(left=0.10, right=0.90, bottom=0.05, top=0.95, hspace=0.35)
    for i in range(len(usecols)):
        ax = fig.add_subplot(len(usecols),1,i+1)
        _plot_heatmap(ax, pd.Series([0]), names[i], 'white')
    plt.savefig(filename)
    plt.close()
    print(filename + "を作成しました")
    return

def _plot_heatmap(ax, series, name, color):
    ax.set_title(name)
    gap = 1 / len(series)
    for i, data in enumerate(series):
        left = i * gap
        right = (i + 1) * gap
        ax.axvspan(left, right, color=color, alpha=data)
    return

def _plot_color_bar(ax, series, name, color_list):
    ax.set_title(name)
    gap = 1 / len(series)
    for i, data in enumerate(series):
        if (not math.isnan(data)):
            left = i * gap
            right = (i + 1) * gap
            ax.axvspan(left, right, color=color_list[int(data)], alpha=0.8)
    return

if __name__ == "__main__":
    dir_path = './'
    target_cow_id_list = ['20113', '20170', '20295', '20299']
    start = datetime.datetime(2018, 10, 1, 0, 0, 0)
    end = datetime.datetime(2018, 10, 30, 0, 0, 0)
    usecols = [4, 5, 6, 7, 8, 9]
    names = ['topic=1', 'topic=2', 'topic=3', 'topic=4', 'topic=5', 'topic_num']
    for cow_id in target_cow_id_list:
        curr_dir = dir_path + cow_id + '/'
        date = start
        while (date < end):
            csv_file = curr_dir + date.strftime('%Y%m%d.csv')
            fig_file = curr_dir + date.strftime('%Y%m%d.png')
            try:
                df = pd.read_csv(csv_file, usecols=usecols, names=names)
                plot_by_day(df, usecols, names, fig_file)
            except pd.errors.ParserError:
                plot_exception(names, fig_file)
            except FileNotFoundError:
                plot_exception(names, fig_file)
            date += datetime.timedelta(days=1)