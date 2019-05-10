#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import statistics
import csv
import re
import datetime
import gc
import sys
import cows.cow as Cow
import cows.geography as geo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import cows.momentum_analysys as ma
import image.adjectory_image as disp
"""
えげつないスパゲッティです
"""
#データベースから指定牛の指定期間のデータを読み込む (元データ (1 Hz) を5s (0.2Hz) に戻す処理も含む)
def read_gps(cow_id, start, end):
	# データを読み込み，それぞれのリストを作成 (情報ごとのリストにするか時間ごとのリストのリストにするかは場合による) 
	time_list = [] # 時間のリスト (主キー)
	position_list = [] # (緯度，経度) のリスト
	distance_list = [] # 距離のリスト
	velocity_list = [] # 速さのリスト
	angle_list = [] # 移動角度のリスト
	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		t_list = []
		pos_list = []
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
					distance, angle = geo.get_distance_and_direction(lat1, lon1, lat2, lon2, False)
					#print(g.get_datetime().strftime("%Y/%m/%d %H:%M:%S") + " : ", lat2 , ",", lon2)
					t_list.append(g.get_datetime()) #時間の格納
					pos_list.append(geo.translate(lat2, lon2)) #位置情報の格納
					dis_list.append(distance) #距離の格納
					vel_list.append(vel2) #速さの格納
					ang_list.append(angle) #角度の格納
				g_before = g
			a = a + datetime.timedelta(minutes = 60)
			del gps_list
			gc.collect()
		time_list.append(t_list) #1日分の時間のリストの格納
		position_list.append(pos_list) #1日分の位置情報の格納
		distance_list.append(dis_list) #1日分の距離のリストの格納
		velocity_list.append(vel_list) #1日分の速さのリストの格納
		angle_list.append(ang_list) #1日分の角度のリストの格納
		del cow
		gc.collect()
		a = dt
	return time_list, position_list, distance_list, velocity_list, angle_list

#データの解析に使用する時間分を抽出する
def select_use_time(t_list, p_list, d_list, v_list, a_list):
	knot = 0.514444 # 1 knot = 0.51444 m/s
	time_tmp_list = []
	position_tmp_list = []
	distance_tmp_list = []
	velocity_tmp_list = []
	angle_tmp_list = []
	for (t, p, d, v, a) in zip(t_list, p_list, d_list, v_list, a_list):
		t = t + datetime.timedelta(hours = 9)
		if(t.hour < 9 or 12 < t.hour):
			time_tmp_list.append(t)
			position_tmp_list.append(p)
			distance_tmp_list.append(d) 
			velocity_tmp_list.append(v * knot) #単位を[m/s]に直しているだけ
			angle_tmp_list.append(a)
	return time_tmp_list, position_tmp_list, distance_tmp_list, velocity_tmp_list, angle_tmp_list
	
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
"""
Parameter
	t_list	:圧縮されていない時間のリスト
	v_list	:圧縮されていない速度のリスト
return
	updated_t_list	:再編した時間のリスト
	updated_v_list	:再編した速度のリスト
	start_list	:休息が始まった時間を格納したリスト
	end_list	:休息が終了した時間を格納したリスト
"""
def zip_rest(t_list, v_list, threshold = 0.069):
	is_rest = False
	tmp_list = [] # 休息時の平均速度を求めるためのリスト
	updated_t_list = [] # 連続した休息を一つに折りたたんで再編した時刻のリスト
	updated_v_list = [] # 連続した休息を一つに折りたたんで再編した速度のリスト
	start_list = [] # 始まりを格納
	end_list = [] # 終わりを格納
	for time, velocity in zip(t_list, v_list):
		if(not(is_rest)):
			# 休息の始まり
			if(velocity < threshold):
				start_list.append(time)
				is_rest = True
				end = time
				tmp_list = []
				tmp_list.append(velocity)
			# 採食or歩行
			else:
				updated_t_list.append(time)
				updated_v_list.append(velocity)
		else:
			# 休息の続き
			if(velocity < threshold):
				tmp_list.append(velocity)
				end = time
			# 休息の終わり
			else:
				end_list.append(end)
				updated_t_list.append(end)
				updated_t_list.append(time)
				updated_v_list.append(sum(tmp_list) / len(tmp_list))
				updated_v_list.append(velocity)
				is_rest = False
	if(len(start_list) == len(end_list) + 1): # 最後が休息だった場合，最後のセグメントを格納する
		end_list.append(t_list[len(t_list) - 1])
		updated_t_list.append(t_list[len(t_list) - 1])
		updated_v_list.append(sum(tmp_list) / len(tmp_list))
	return updated_t_list, updated_v_list, start_list, end_list

#重心を求める (休息時の座標の重心を求めたい．また，休息していた時間も合わせた形で返したい)
"""
Parameter
	t_list	:時間のリスト
	p_list	:位置情報のリスト (latitude, longitude)
	s_list, e_list	:それぞれ始まりと終わりを格納したリスト
return
	各状態の重心とその時間
"""
def determin_center(t_list, p_list, s_list, e_list):
	center_list = []
	for start, end in zip(s_list, e_list):
		lats = []
		lons = []
		for i, time in enumerate(t_list):
			if (start <= time and time <= end) :
				lats.append(p_list[i][0])
				lons.append(p_list[i][1])
			elif (end < time) :
				break
		if(len(lats) != 0 and len(lons) != 0):
			lat = sum(lats) / len(lats)
			lon = sum(lons) / len(lons)
			center_list.append((lat, lon, (end - start).total_seconds() / 60)) # (lat, lon, 滞在時間 [minutes])
	return center_list


#圧縮した休息に合わせて距離・角度・位置のリストを再編する
"""
Parameter
	t_list	:圧縮されていない時間のリスト
	p_list	:圧縮されていない位置情報のリスト
	d_list	:圧縮されていない距離のリスト
	a_list	:圧縮されていない角度のリスト
	s_list, e_list	:それぞれ休息の始まりと終わりを格納したリスト
	zipped_g_list	:圧縮された位置情報のリスト(lat, lon 滞在時間)を含むdetermin_centerで求めています
return
	updated_zipped_list
"""
def remake_gps_list(t_list, p_list, d_list, a_list, start_list, end_list, zipped_g_list, zipped_v_list):
	updated_zipped_list = []
	index = 0
	previous_rest_end = t_list[0] #以前の休息の終了時刻
	rest_start = start_list[index] #休息の開始時刻
	rest_end = end_list[index] #休息の終了時刻
	first_start = start_list[0]
	last_end = end_list[len(end_list) - 1]
	for time, p, d, a in zip(t_list, p_list, d_list, a_list):
		if(rest_end < time):
			index += 1
			if(len(start_list) <= index):
				index -= 1 #index固定 (breakしてはいけないので)
			else:
				previous_rest_end = end_list[index - 1]
				rest_start = start_list[index]
				rest_end = end_list[index]
		#休息中でないところには通常通り登録する
		if((previous_rest_end < time and time < rest_start) or (time < first_start or last_end < time)):
			updated_zipped_list.append((time, p, d, a))

	return updated_zipped_list

#特徴をCSVにして出力する (圧縮が既に行われている前提) 
"""
Parameter
	t_list	:時間のリスト
	p_list	:位置情報のリスト
	s_list, e_list	:それぞれ始まりと終わりを格納したリスト
"""
def output_feature_info(t_list, p_list, s_list, e_list):
	is_first = True
	feature_list =[]
	
	#####登録情報#####
	previous_lat = 0.0
	previous_lon = 0.0
	lat = 0.0
	lon = 0.0
	previous_rest_time = None #前の休息の時間
	previous_rest_end = None #前の休息の終わりの時刻
	rest_time = None #現在の休息の時間
	rest_start = None #現在の休息の始まりの時間
	moving_distance = None #休息間の距離
	moving_direction = None #次の休息への移動方向
	interval_between_rest = None #休息間の時間間隔

	#####登録#####
	g_list = determin_center(t_list, p_list, s_list, e_list) # 休息の重心を求める
	for start, end, pos in zip(s_list, e_list, g_list):
		if (is_first) :
			previous_lat = pos[0]
			previous_lon = pos[1]
			previous_rest_time = pos[2]
			previous_rest_end = end
			is_first = False
		else:
			#if (pos[2] > 1) : # 1分以上の休息に対して
				lat = pos[0]
				lon = pos[1]
				rest_time = pos[2]
				rest_start = start
				moving_distance, moving_direction = geo.get_distance_and_direction(previous_lat, previous_lon, lat, lon, True) #休息間の距離
				interval_between_rest = (rest_start - previous_rest_end).total_seconds() / 60 #休息間の時間間隔 [minutes]
				sum_distance = 0.0 #休息間の道のり(各サンプリング区間の距離の総和)

				###登録###
				feature_list.append([previous_rest_end, previous_lat, previous_lon, previous_rest_time, rest_start, lat, lon, rest_time, moving_distance, moving_direction, interval_between_rest])
				###引継###
				previous_lat = lat
				previous_lon = lon
				previous_rest_time = rest_time
				previous_rest_end = end

	#####出力#####
	with open("行動解析/feature.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(("Last end time", "Last latitude", "Last longitude", "last continuous time", "Start time", "Latitude", "Longitude", "Continuous time", "Moving distance", "Moving direction", "Interval between rests"))
		for feature in feature_list:
			writer.writerow(feature)
	return


#移動速度に応じて分類する
"""
Parameter
	v_list	:速さのリスト
return
	それぞれが振り分けられたクラスタ ("red", "green", "blue")
"""
def calassify_distance(v_list, graze = 0.069, walk = 0.18):
	data_list = []
	for data in v_list:
		if(data < graze):
			data_list.append("red")
		elif(data < walk):
			data_list.append("green")
		else:
			data_list.append("blue")
	return data_list

if __name__ == '__main__':
	start = datetime.datetime(2018, 2, 22, 0, 0, 0)
	end = datetime.datetime(2018, 2, 23, 0, 0, 0)
	time_list, position_list, distance_list, velocity_list, angle_list = read_gps(20283, start, end) #2次元リスト (1日分 * 日数分)
	for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
		print(len(t_list))
		t_list, p_list, d_list, v_list, a_list = select_use_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
		#t_list, d_list, a_list = ma.convo_per_minutes(t_list, d_list, a_list, 3) #畳み込み
		
		#c_list = calassify_distance(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
		#scatter_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示

		zipped_t_list, zipped_v_list, s_list, e_list = zip_rest(t_list, v_list) # 休息を圧縮する
		g_list = determin_center(t_list, p_list, s_list, e_list) # 休息の重心を求める
		test = remake_gps_list(t_list, p_list, d_list, a_list, s_list, e_list, g_list, zipped_v_list) #圧縮された休息を用いて再度リストを作り直す
		print(len(g_list))
		print(len(test))
		print(len(zipped_t_list))
		output_feature_info(t_list, p_list, s_list, e_list) # 特徴を出力する
		#display = disp.Adjectory(True)
		#display.plot_moving_ad(p_list) # 移動の軌跡をプロット
		#display = disp.Adjectory(True)
		#display.plot_rest_place(g_list) # 休息の場所の分布のプロット

		c_list = calassify_distance(zipped_v_list) # クラスタ分けを行う (速さを3つに分類しているだけ)
		scatter_plot(zipped_t_list, zipped_v_list, c_list) # 時系列で速さの散布図を表示
