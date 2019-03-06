#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import LocalKalman
import csv

#csv読み込み#1行目ヘッダー (移動平均後)
r_filename = "../csv/gps/weight_m.csv"
t_filename = "../csv/gps/weight_t.csv"
w_filename = "../csv/gps/weight_a.csv"
df = pd.read_csv(filepath_or_buffer = r_filename, index_col = 0, encoding = "utf-8", sep = ",", header = 0)
df = df.fillna(0)
header = ["TIME"] + list(df.head(0))


s1 = datetime.datetime(2018, 5, 15)
e1 = datetime.datetime(2018, 6, 15)
s2 = datetime.datetime(2018, 10, 1)
e2 = datetime.datetime(2018, 11, 1)
t_row_list1 = []
t_row_list2 = []
t_time_list = []
#訓練データの作成
for index, row in df.iterrows():
        dt = datetime.datetime.strptime(index, "%Y/%m/%d %H:%M")
        if(s1 <= dt and dt < e1):
                t_time_list.append(index)
                t_row_list1.append(list(row))
        elif(s2 <= dt and dt < e2):
                t_time_list.append(index)
                t_row_list2.append(list(row))
        elif(e2 <= dt):
                break
"""
#訓練データの作成
N = 144 * 30
t_row_list2 = []
for i in range(len(df.columns)):
        t_row_list1 = []
        cosines = list(df.iloc[4254:, i])
        for cos in cosines:
                if(len(t_row_list1) < N):
                        if(cos != 0):
                                t_row_list1.append(cos)       
                else:
                        break
        t_row_list2.append(t_row_list1)
"""
"""
with open(t_filename, mode = "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        t_row_list = t_row_list1 + t_row_list2
        for t, r in zip(t_time_list, t_row_list):
                writer.writerow([t] + r)
"""
dt_list = []
abnormal_scores_list = [] #異常値の各牛のリストのリスト
#datetime型に変換
for str in df.index:
        dt_list.append(datetime.datetime.strptime(str, "%Y/%m/%d %H:%M"))
dt_list = dt_list[4254:]
train_data_list = (np.array(t_row_list1).T).tolist()#(np.array(t_row_list1 + t_row_list2).T).tolist()
#各行に対して行う
for i in range(len(df.columns)):
        cosines = list(df.iloc[4254:, i])
        print(i)
        train_data = train_data_list[i]#[data for data in train_data_list[i] if(data != 0)]
        k = LocalKalman.KalmanFilter(0.25, 0.25, train_data)
        ab_scores, x_scores, p_scores, k_scores = k.sequentialProcess(cosines)
        abnormal_scores_list.append(ab_scores)
        #plot関係の設定
        start = datetime.datetime(2018, 7, 1)
        end = datetime.datetime(2018, 7, 15)                                                            
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
        ax1.plot(dt_list, cosines, 'b')
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
        ax5.set_ylim(0, 5.0) #表示範囲を設定
        ax5.plot(dt_list, ab_scores, 'r')
        ax5.legend(("Score",), loc='upper left')

        #縦になったキャプションが被るのを防止
        plt.subplots_adjust(left = 0.05, bottom = 0.08, right = 0.95, top = 0.92, hspace=1.0)

        plt.show()

tmp_array = np.array(abnormal_scores_list)
abnormal_scores_list = (tmp_array.T).tolist()
with open(w_filename, mode = "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for time, row in zip(dt_list, abnormal_scores_list):
                writer.writerow([time.strftime("%Y/%m/%d %H:%M:%S")] + row)

"""
#plot関係の設定
start = dt_list[0]
end = dt_list[len(dt_list) - 1]
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
ax1.plot(dt_list, cosines, 'b')
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
#plt.show()
