#-*- encoding:utf-8 -*-
import numpy as np
import ssm
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import LocalKalman
import sys

#各月の変換表
month = {"Jan":"1", "Feb":"2", "Mar":"3", "Apr":"4", "May":"5", "Jun":"6", "Jul":"7", "Aug":"8", "Sep":"9", "Oct":"10", "Nov":"11", "Dec":"12"}
#csv読み込み#1行目ヘッダー (移動平均後)
df = pd.read_csv(filepath_or_buffer = "../Data/0.5/cosineIntegrate0.5.csv", encoding = "utf-8", sep = ",", header = 0, usecols = [0, int(sys.argv[1])], names = ('date', 'value'))
df = df.fillna(0)
dt_list = [] #日付のリスト
data_list = [] #データのリスト
#データをリストに格納
for data in df['value']:
	data_list.append(data)
	
#datetime型に変換
for str in df['date']:
    cal = month[str[4:7]] + str[8:]
    dt = datetime.datetime.strptime(cal, '%m%d %H:%M:%S JST %Y')
    dt_list.append(dt)

"""
        Parameters
        ----------
        N		:int	:データの使用個数 (本書での2Nであることに注意)
        w    	: int	:Window size
"""
#使用個数を抽出
N = 1440
w = 144
train_data_list = data_list[:N]
train_dt_list = dt_list[:N]

#X, Z = ssm.estimateStateSeries(train_data_list, w)

k = LocalKalman.KalmanFilter(0.01, 0.01, train_data_list)
t_ab_scores, t_x_scores, t_p_scores, t_k_scores = k.sequentialProcess(train_data_list)
ab_scores, x_scores, p_scores, k_scores = k.sequentialProcess(data_list)



#plot関係の設定
start = datetime.datetime(2018, 8, 1, 0, 0, 0)
end = datetime.datetime(2018, 10, 31, 1, 0, 0)
t = start
date = []
while t <= end:
	date.append(t)
	t = t + datetime.timedelta(days = 1)
fig = plt.figure()
ax1 = fig.add_subplot(5,1,1) #4行1列の図
ax2 = fig.add_subplot(5, 1, 2) #4行1列の図
ax3 = fig.add_subplot(5, 1, 3) #4行1列の図
ax4 = fig.add_subplot(5, 1, 4) #4行1列の図
ax5 = fig.add_subplot(5, 1, 5) #4行1列の図
#ax2 = ax1.twinx() #グラフを重ねる
#ax1の設定
ax1.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax1.set_xlim(start, end) #表示範囲を設定
#ax1.set_ylim(-0.05, 0.05) #表示範囲を設定
ax1.plot(dt_list, data_list, 'b')
ax1.legend(("Observe",), loc='upper left')

#ax2の設定
ax2.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax2.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax2.set_xlim(start, end) #表示範囲を設定
#ax2.set_ylim(0, 0.03) #表示範囲を設定
ax2.plot(dt_list, x_scores, 'r')
ax2.legend(("State",), loc='upper left')

#ax3の設定
ax3.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax3.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax3.set_xlim(start, end) #表示範囲を設定
#ax3.set_ylim(-0.03, 0.03) #表示範囲を設定
ax3.plot(dt_list, p_scores, 'r')
ax3.legend(("Varience",), loc='upper left')

#ax4の設定
ax4.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax4.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax4.set_xlim(start, end) #表示範囲を設定
#ax4.set_ylim(-0.03, 0.03) #表示範囲を設定
ax4.plot(dt_list, k_scores, 'r')
ax4.legend(("Gain",), loc='upper left')

#ax5の設定
ax5.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax5.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax5.set_xlim(start, end) #表示範囲を設定
ax5.set_ylim(0, 1.0) #表示範囲を設定
ax5.plot(dt_list, ab_scores, 'r')
ax5.legend(("Score",), loc='upper left')

#縦になったキャプションが被るのを防止
plt.subplots_adjust(left = 0.05, bottom = 0.08, right = 0.95, top = 0.92, hspace=1.0)

plt.show()
"""

#plot関係の設定
start = datetime.datetime(2018, 8, 1)
end = datetime.datetime(2018, 8, 10)
t = start
date = []
while t <= end:
	date.append(t)
	t = t + datetime.timedelta(days = 1)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1) #4行1列の図
#ax2 = fig.add_subplot(4, 1, 2) #4行1列の図
#ax3 = fig.add_subplot(4, 1, 3) #4行1列の図
#ax4 = fig.add_subplot(4, 1, 4) #4行1列の図
ax2 = ax1.twinx() #グラフを重ねる
#ax1の設定
ax1.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax1.set_xlim(start, end) #表示範囲を設定
ax1.set_ylim(-0.01, 0.01) #表示範囲を設定
ax1.plot(train_dt_list, train_data_list, 'b')

#ax2の設定
ax2.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax2.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax2.set_xlim(start, end) #表示範囲を設定
ax2.set_ylim(-0.01, 0.01) #表示範囲を設定
ax2.plot(train_dt_list, t_x_scores, 'r')

"""
"""
#ax3の設定
ax3.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax3.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax3.set_xlim(start, end) #表示範囲を設定
#ax3.set_ylim(-0.03, 0.03) #表示範囲を設定
ax3.plot(train_dt_list, t_p_scores, 'r')

#ax4の設定
ax4.set_xticklabels(date, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
ax4.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")) #日付の表示形式を決定
ax4.set_xlim(start, end) #表示範囲を設定
#ax4.set_ylim(-0.03, 0.03) #表示範囲を設定
ax4.plot(train_dt_list, t_k_scores, 'r')

"""
plt.show()
