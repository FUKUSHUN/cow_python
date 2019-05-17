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

#休息時の速度を一つに折りたたむ
"""
Parameter
	t_list	:圧縮されていない時間のリスト
	v_list	:圧縮されていない速度のリスト
return
	start_list	:休息が始まった時間を格納したリスト
	end_list	:休息が終了した時間を格納したリスト
"""
def zip_rest(t_list, v_list, threshold = 0.069):
	print(sys._getframe().f_code.co_name, "実行中")
	print("休息を圧縮します---")
	is_rest = False
	start_list = [] # 始まりを格納
	end_list = [] # 終わりを格納
	for time, velocity in zip(t_list, v_list):
		if(not(is_rest)):
			# 休息の始まり
			if(velocity < threshold):
				start_list.append(time)
				is_rest = True
				end = time
		else:
			# 休息の続き
			if(velocity < threshold):
				end = time
			# 休息の終わり
			else:
				end_list.append(end)
				is_rest = False
	if(len(start_list) == len(end_list) + 1): # 最後が休息だった場合，最後のセグメントを格納する
		end_list.append(t_list[len(t_list) - 1])
	print("---休息を圧縮しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return start_list, end_list


#休息時間と休息の重心（緯度・経度）を時間順に格納したリストを返す
"""
Parameter
	p_list	:圧縮後の位置情報のリスト (latitude, longitude). すでに重心は計算済み
	start_list, end_list	:それぞれ始まりと終わりの時刻を格納したリスト
return
	各状態の重心とその時間
"""
def make_rest_data(p_list, start_list, end_list):
	print(sys._getframe().f_code.co_name, "実行中")
	print("休息時間と休息の重心を時刻順にリストに格納します---")
	i = 0
	center_list = []
	if (len(p_list) == len(start_list) and len(start_list) == len(end_list)):
		for start, end in zip(start_list, end_list):
			center_list.append((p_list[i][0], p_list[i][1], (end - start).total_seconds() / 60)) # (lat, lon, 滞在時間 [minutes])
			i += 1
		print("---休息時間と休息の重心を時刻順にリストに格納しました")
		print(sys._getframe().f_code.co_name, "正常終了\n")
	else:
		print("---error: リスト長が違います．正しく指定してください---")
		print("p_list:",len(p_list), " start_list:",len(start_list), " end_list:",len(end_list))
		print(sys._getframe().f_code.co_name, "異常終了\n")
	return center_list


#圧縮した休息に合わせて距離・角度・位置のリストを再編する
"""
Parameter
	t_list	:圧縮されていない時間のリスト
	p_list	:圧縮されていない位置情報のリスト
	d_list	:圧縮されていない距離のリスト
	v_list	:圧縮されていない速さのリスト
	a_list	:圧縮されていない角度のリスト
	start_list, end_list	:それぞれ休息の始まりと終わりを格納したリスト
return
	updated_zipped_list	圧縮した休息に合わせてリメイクしたデータリスト
	updated_zipped_rest_list	圧縮した休息のみを抽出したリスト，圧縮時に重心や平均速度，総距離などを求めています
"""
def remake_gps_list(t_list, p_list, d_list, v_list, a_list, start_list, end_list):
	print(sys._getframe().f_code.co_name, "実行中")
	updated_zipped_list = []
	updated_zipped_rest_list = []
	index = 0
	print("休息中でないところのリストをまず作成します---")
	###走査用変数###
	previous_rest_end = t_list[0] #以前の休息の終了時刻
	rest_start = start_list[index] #休息の開始時刻
	rest_end = end_list[index] #休息の終了時刻
	first_start = start_list[0]
	last_end = end_list[len(end_list) - 1]
	###時間順に辿っていき休息でないところの情報を格納していく###
	for time, p, d, v, a in zip(t_list, p_list, d_list, v_list, a_list):
		if(rest_end < time):
			index += 1
			if(len(start_list) <= index):
				index -= 1 #index固定 (breakしてはいけないので)
			else:
				previous_rest_end = end_list[index - 1]
				rest_start = start_list[index]
				rest_end = end_list[index]
		#休息中でないところにはそのまま1データずつ登録する
		if((previous_rest_end < time and time < rest_start) or (time < first_start or last_end < time)):
			updated_zipped_list.append((time, p, d, v, a))
	print("---休息中でないところのリストが完成しました")
	print("登録数:", len(updated_zipped_list))

	index = 0
	
	print("圧縮された休息のリストを作成します---")
	###走査用変数###
	rest_start = start_list[index] #休息の開始時刻
	rest_end = end_list[index] #休息の終了時刻
	###演算用変数###
	ave_lat_list = [] #平均を求めるための緯度のリスト
	ave_lon_list = [] #平均を求めるための経度のリスト
	sum_dis_list = [] #休息中の距離の総和（おそらく休息中の移動距離の総和は0）を求めるための距離のリスト
	ave_vel_list = [] #休息中の速さの平均を求めるための速さのリスト
	###時間順に辿っていき休息でないところの情報を格納していく###
	for time, p, d, v, a in zip(t_list, p_list, d_list, v_list, a_list):
		if(rest_end < time):
			index += 1
			if(len(start_list) <= index):
				break
			else:
				rest_start = start_list[index]
				rest_end = end_list[index]
				ave_lat_list = []
				ave_lon_list = []
				sum_dis_list = []
				ave_vel_list = []
		#休息中のまとまりごとに重心（平均）を求めて登録する. 休息中の角度はあまり必要ないかな．だから0にする
		if(rest_start <= time and time <= rest_end):
			ave_lat_list.append(p[0])
			ave_lon_list.append(p[1])
			sum_dis_list.append(d)
			ave_vel_list.append(v)
			#登録が最後の時は平均や総和を求めてリストに登録する
			if(time == rest_end):
				lat = sum(ave_lat_list) / len(ave_lat_list)
				lon = sum(ave_lon_list) / len(ave_lon_list)
				dis = sum(sum_dis_list)
				vel = sum(ave_vel_list) / len(ave_vel_list)
				updated_zipped_rest_list.append((rest_start, (lat, lon), dis, vel, 0.0)) #時間は休息の始まりに設定（必ずstart_list, end_listをキーに使うこと）
	print("---休息中でないところのリストが完成しました")
	print("登録数:",len(updated_zipped_rest_list))
	updated_zipped_list.extend(updated_zipped_rest_list) #休息とそれ以外のリストの結合
	updated_zipped_list.sort(key=lambda x: x[0]) #リストを時刻順にソート
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return updated_zipped_list, updated_zipped_rest_list

#特徴をCSVにして出力する (圧縮が既に行われている前提) 
"""
Parameter
	t_list	:圧縮後の時刻のリスト
	p_list	:圧縮後の位置情報のリスト
	d_list	:圧縮後の距離のリスト
	start_list, end_list	:それぞれ休息の始まりと終わりの時刻を格納したリスト
"""
def output_feature_info(t_list, p_list, d_list, start_list, end_list):
	print(sys._getframe().f_code.co_name, "実行中")
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

	print("特徴を計算します---")
	#####登録#####
	g_list = make_rest_data(p_list, start_list, end_list) # 休息の重心を求める
	for start, end, pos in zip(start_list, end_list, g_list):
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
			sum_of_distance = calc_sum_of_dis(t_list, d_list, previous_rest_end, rest_start) #休息間の道のり(各サンプリング区間の距離の総和)

			###登録###
			feature_list.append([previous_rest_end, previous_lat, previous_lon, previous_rest_time, rest_start, lat, lon, rest_time, moving_distance, sum_of_distance, moving_direction, interval_between_rest])
			###引継###
			previous_lat = lat
			previous_lon = lon
			previous_rest_time = rest_time
			previous_rest_end = end
	print("---特徴を計算しました")

	filename = "行動解析/feature.csv"
	print(filename + "に出力します---")
	#####出力#####
	with open(filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(("Last end time", "Last latitude", "Last longitude", "last continuous time", "Start time", "Latitude", "Longitude", "Continuous time", "Moving distance", "Moving amount", "Moving direction", "Interval between rests"))
		for feature in feature_list:
			writer.writerow(feature)
	print("---" + filename + "に出力しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return

#休息間の総移動距離を求める
"""
Paramter
	d_list	:圧縮後の距離のリスト
	start, end	:それぞれ休息の始まりと終わりの時刻 (前の休息の終わりがstart, 次の休息の始まりがendであることに注意)
"""
def calc_sum_of_dis(t_list, d_list, start, end):
	sum_of_distance = 0.0
	for t, d in zip(t_list, d_list):
		if (start < t and t <= end):
			sum_of_distance += d
		elif (end <= t):
			break
	return sum_of_distance

#移動速度に対して分類を行い視覚化を可能にする
"""
Parameter
	v_list	:速さのリスト
	l_list	:ラベルのリスト．Noneであれば速さだけで分類を行い，Noneでなければこのラベルを元に分類を行う
return
	それぞれが振り分けられたクラスタ ("red", "green", "blue")
"""
def calassify_velocity(v_list, graze = 0.069, walk = 0.18, l_list=None):
	data_list = []
	if (l_list is None):
		for data in v_list:
			if(data < graze):
				data_list.append("red")
			elif(data < walk):
				data_list.append("green")
			else:
				data_list.append("blue")
	else:
		for data, label in zip(v_list, l_list):
			if (label == "R"):
				data_list.append("red") # Rest
			elif (label == "G"):
				data_list.append("green") # Graze
			elif (label == "W"):
				data_list.append("blue") # Walk
			elif (label == "O"):
				data_list.append("yellow") # Other
			elif (label == "N"):
				data_list.append("magenta") # None
			else:
				data_list.append("black") #なぜかどれにも当てはまらない(デバッグ用)
	return data_list

#圧縮した休息とその他をラベル付きで解凍する
"""
Parameter
	zipped_t_list	:圧縮された時間のリスト
	zipped_l_list	:圧縮により求められたラベルのリスト
	start_list, end_list	:それぞれ休息の始まりと終わりの時刻を格納したリスト
	t_list	:圧縮前の時間のリスト（いつも5sの間隔で作られている訳ではないので元の時間のリストの時間を参照する）
Return
	behavior_list	:解凍したリスト
"""
def decode(zipped_t_list, zipped_l_list, start_list, end_list, t_list):
	print(sys._getframe().f_code.co_name, "実行中")
	label = None
	behavior_list = []
	index = 0
	rest_start = start_list[index] #休息の開始時刻
	rest_end = end_list[index] #休息の終了時刻
	for time in t_list:
		if(rest_end < time):
			index += 1
			if(len(start_list) <= index):
				index -= 1 #index固定 (breakしてはいけないので)
			else:
				rest_start = start_list[index]
				rest_end = end_list[index]
		if (rest_start <= time and time <= rest_end):
			label = "R" # Rest
		else:
			for t, l in zip(zipped_t_list, zipped_l_list):
				if(t == time):
					label = l
					break
		if (label is not None):
			behavior_list.append((time, label))
		else:
			print("ラベルがありません")
			sys.exit() # 訓練データなどでラベルがない場合でもzipped_l_listにはあらかじめ"N"が登録されているものとする
		label = None #間違って前のラベルを引き継がないように
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return behavior_list

if __name__ == '__main__':
	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
	time_list, position_list, distance_list, velocity_list, angle_list = read_gps(20158, start, end) #2次元リスト (1日分 * 日数分)
	for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
		print(len(t_list))
		t_list, p_list, d_list, v_list, a_list = select_use_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
		#t_list, d_list, a_list = ma.convo_per_minutes(t_list, d_list, a_list, 3) #畳み込み
		
		c_list = calassify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
		scatter_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示

		s_list, e_list = zip_rest(t_list, v_list) # 休息を圧縮する
		zipped_list, zipped_rest_list = remake_gps_list(t_list, p_list, d_list, v_list, a_list, s_list, e_list) #圧縮された休息を用いて再度リストを作り直す
		g_list = make_rest_data([row[1] for row in zipped_rest_list], s_list, e_list) # 休息時間と重心だけのリストにする
		#output_feature_info([row[0] for row in zipped_list], [row[1] for row in zipped_rest_list], [row[2] for row in zipped_list], s_list, e_list) # 特徴を出力する
		#display = disp.Adjectory(False)
		#display.plot_moving_ad(p_list) # 移動の軌跡をプロット
		display = disp.Adjectory(True)
		display.plot_rest_place(g_list) # 休息の場所の分布のプロット

		#c_list = calassify_velocity([row[3] for row in zipped_list]) # クラスタ分けを行う (速さを3つに分類しているだけ)
		#scatter_plot([row[0] for row in zipped_list], [row[3] for row in zipped_list], c_list) # 時系列で速さの散布図を表示

