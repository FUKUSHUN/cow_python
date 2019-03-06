#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import statistics
import re
import datetime
import gc
import sys
import cows.cow as Cow
import cows.geography as geo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cows.momentum_analysys as ma
import image.adjectory_image as disp

#データベースから指定牛の指定期間のデータを読み込む (元データ (1 Hz) を5sごとに戻す処理も含む)
def read_gps(cow_id, start, end):
	time_list = []
	distance_list = []
	velocity_list = []
	angle_list = []
	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		t_list = []
		dis_list = []
		vel_list = []
		ang_list = []
		cow = Cow.Cow(cow_id, dt)
		dt = dt + datetime.timedelta(days = 1)
		while(a <= dt and a < end):
			gps_list = cow.get_gps_list(a, a + datetime.timedelta(minutes = 60))
			g_before = None
			for i in range(int(len(gps_list) / 5)):
				g = gps_list[i * 5]
				if g_before is not None:
					lat1, lon1, vel1 = g_before.get_gps_info(g_before.get_datetime())
					lat2, lon2, vel2 = g.get_gps_info(g.get_datetime())
					distance, angle = geo.get_distance_and_direction(lat1, lon1, lat2, lon2)
					#print(g.get_datetime().strftime("%Y/%m/%d %H:%M:%S") + " : ", lat2 , ",", lon2)
					t_list.append(g.get_datetime()) #時間の格納
					dis_list.append(distance) #距離の格納
					vel_list.append(vel2) #速さの格納
					ang_list.append(angle) #角度の格納
				g_before = g
			a = a + datetime.timedelta(minutes = 60)
			del gps_list
			gc.collect()
		time_list.append(t_list) #1日分の時間のリストの格納
		distance_list.append(dis_list) #1日分の距離のリストの格納
		velocity_list.append(vel_list) #1日分の速さのリストの格納
		angle_list.append(ang_list) #1日分の角度のリストの格納
		del cow
		gc.collect()
		a = dt
	return time_list, distance_list, velocity_list, angle_list

#データの解析に使用する時間分を抽出する
def select_use_time(t_list, d_list, v_list, a_list):
	knot = 0.514444 # 1 knot = 0.51444 m/s
	time_tmp_list = []
	distance_tmp_list = []
	velocity_tmp_list = []
	angle_tmp_list = []
	for (t, d, v, a) in zip(t_list, d_list, v_list, a_list):
		t = t + datetime.timedelta(hours = 9)
		if(t.hour < 18 and 15 < t.hour):
			time_tmp_list.append(t)
			distance_tmp_list.append(d) 
			velocity_tmp_list.append(v * knot) #単位を[m/s]に直しているだけ
			angle_tmp_list.append(a)
	return time_tmp_list, distance_tmp_list, velocity_tmp_list, angle_tmp_list
	
#プロットする
def plot_distance_data(t_list, d_list, a_list):
	#グラフ化のための設定
	fig = plt.figure()
	ax1 = fig.add_subplot(2,1,1) #2行1列の図
	ax2 = fig.add_subplot(2,1,2) #2行1列の図
	#ax1の設定
	ax1.set_xticklabels(t_list, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
	ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
	ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H-%M")) #日付の表示形式を決定
	ax1.plot(t_list, d_list, 'b')
	ax1.legend(("Distance",), loc='upper left')
	
	#ax2の設定
	ax2.set_xticklabels(t_list, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
	ax2.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
	ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H-%M")) #日付の表示形式を決定
	ax2.plot(t_list, a_list, 'r')
	ax2.legend(("Angle",), loc='upper left')
	#plt.savefig(t_list[0].strftime("%y-%m-%d") + "-test.png")
	plt.show()

#プロットする
def plot_velocity_data(t_list, v_list):
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

#散布図形式でプロットする
def scatter_plot(t_list, d_list, c_list):
	x_list = []
	for i in range(len(t_list)):
		x_list.append(i)
	x = np.array(x_list)
	y = np.array(d_list)
	c = np.array(c_list)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x, y, c = c, s = 1)
	plt.show()

#休息時の速度を一つに折りたたむ
def zip_rest(t_list, v_list, threshold = 0.069):
	lock = False
	tmp_list = [] # 休息時の平均速度を求めるためのリスト 
	tmp_t_list = [] # 連続した休息を一つに折りたたんで再編した時刻のリスト
	tmp_v_list = [] # 連続した休息を一つに折りたたんで再編した速度のリスト
	start_list = [] # 始まりを格納 (未使用)
	end_list = [] # 終わりを格納 (未使用)
	for time, velocity in zip(t_list, v_list):
		if(not(lock)):
			if(velocity < threshold):
				start_list.append(time)
				lock = True
				tmp_list = []
				tmp_list.append(velocity)
			else:
				tmp_t_list.append(time)
				tmp_v_list.append(velocity)
		else:
			if(velocity < threshold):
				tmp_list.append(velocity)
				end = time
			else:
				end_list.append(end)
				tmp_t_list.append(end)
				tmp_t_list.append(time)
				tmp_v_list.append(sum(tmp_list) / len(tmp_list))
				tmp_v_list.append(velocity)
				lock = False
	return tmp_t_list, tmp_v_list

#移動距離に応じて分類する
def calassify_distance(d_list, graze = 0.069, walk = 0.18):
	data_list = []
	for data in d_list:
		if(data < graze):
			data_list.append("red")
		elif(data < walk):
			data_list.append("green")
		else:
			data_list.append("blue")
	return data_list

if __name__ == '__main__':
	display = disp.Adjectory(False)
	display.write()
	start = datetime.datetime(2018, 12, 22, 0, 0, 0)
	end = datetime.datetime(2018, 12, 23, 0, 0, 0)
	time_list, distance_list, velocity_list, angle_list = read_gps(20261, start, end) #2次元リスト (1日分 * 日数分)
	for (t_list, d_list, v_list, a_list) in zip(time_list, distance_list, velocity_list, angle_list):
		print(len(t_list))
		t_list, d_list, v_list, a_list = select_use_time(t_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
		#print(len(t_list))
		#t_list, d_list, a_list = ma.convo_per_minutes(t_list, d_list, a_list, 3)
		t_list, v_list = zip_rest(t_list, v_list)
		c_list = calassify_distance(v_list)
		scatter_plot(t_list, v_list, c_list)
		#print(len(t_list))
		#plot_distance_data(t_list, d_list, a_list) #plot
