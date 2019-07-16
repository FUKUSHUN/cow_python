#-*- encoding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import statistics
import csv
import datetime
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import sklearn.decomposition as skd
import sklearn
import analyze_main.hmm as hmm

import behavior_classification.loading as loading
import behavior_classification.preprocessing as preprocessing
import cows.cow as Cow
import cows.geography as geo
import cows.momentum_analysys as ma
import image.adjectory_image as disp

"""
えげつないスパゲッティです
"""

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

#圧縮操作を行う
"""
Parameter
	t_list	: 圧縮されていない時間のリスト
	p_list	: 圧縮されていない(緯度, 経度)のリスト
	d_list	: 圧縮されていない距離のリスト
	v_list	: 圧縮されていない速度のリスト
return
	zipped_list	: 各要素が (0: (start, end), 1: 重心 (緯度, 経度), 2: 総距離, 3: 平均速度, 4: 暫定的なラベル) の形のリスト
"""
def zipping(t_list, p_list, d_list, v_list):
	print(sys._getframe().f_code.co_name, "実行中")
	print("圧縮を開始します---")
	zipped_list = []
	p_tmp_list = [] # 場所 (緯度, 軽度) 用
	d_tmp_list = [] # 距離用
	v_tmp_list = [] # 速度用
	is_rest = False
	is_graze = False
	is_walk = False

	start = None #未定義エラー除去
	end = None 
	for time, place, distance, velocity in zip(t_list, p_list, d_list, v_list):
		if (is_rest): # 1個前が休息
			if (choice_state(velocity) == 0):
				p_tmp_list.append(place)
				d_tmp_list.append(distance)
				v_tmp_list.append(velocity)
				end = time
			else: # 休息の終了
				zipped_list.append(((start, end), p_tmp_list, d_tmp_list, v_tmp_list, 0))
				p_tmp_list = [place]
				d_tmp_list = [distance]
				v_tmp_list = [velocity]
				start = time
				end = time
		elif (is_graze): # 1個前が採食
			if (choice_state(velocity) == 1):
				p_tmp_list.append(place)
				d_tmp_list.append(distance)
				v_tmp_list.append(velocity)
				end = time
			else: # 採食の終了
				zipped_list.append(((start, end), p_tmp_list, d_tmp_list, v_tmp_list, 1))
				p_tmp_list = [place]
				d_tmp_list = [distance]
				v_tmp_list = [velocity]
				start = time
				end = time
		elif (is_walk): # 1個前が歩行
			if (choice_state(velocity) == 2):
				p_tmp_list.append(place)
				d_tmp_list.append(distance)
				v_tmp_list.append(velocity)
				end = time
			else: # 歩行の終了
				zipped_list.append(((start, end), p_tmp_list, d_tmp_list, v_tmp_list, 2))
				p_tmp_list = [place]
				d_tmp_list = [distance]
				v_tmp_list = [velocity]
				start = time
				end = time
		else: # ループの一番最初だけここ
			start = time
			end = time
			p_tmp_list.append(place)
			d_tmp_list = [distance]
			v_tmp_list = [velocity]

		if (choice_state(velocity) == 0):
			is_rest = True
			is_graze = False
			is_walk = False
		elif (choice_state(velocity) == 1):
			is_rest = False
			is_graze = True
			is_walk = False
		else:
			is_rest = False
			is_graze = False
			is_walk = True
	
	# 最後の行動を登録して登録終了
	zipped_list.append(((start, end), p_tmp_list, d_tmp_list, v_tmp_list, choice_state(velocity)))
	print("---圧縮が終了しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return zipped_list

#休息・採食・歩行を判断する（今は閾値や最近傍のアプローチだが変更する可能性あり）
def choice_state(velocity, r_threshold = 0.0694, g_threshold = 0.181):
	if (velocity < r_threshold):
		return 0 # 休息
	elif (r_threshold <= velocity and velocity < g_threshold):
		return 1 # 採食
	else:
		return 1 # 歩行 (実施不透明)

#休息時間と休息の重心（緯度・経度）を時刻順に格納したリストを返す
"""
Parameter
	zipped_list		: 圧縮後のリスト (各要素の１番目が (start, end), 2番目が (緯度, 経度))
return
	rest_list	: 休息の重心とその時間
"""
def make_rest_data(zipped_list):
	print(sys._getframe().f_code.co_name, "実行中")
	print("休息時間と休息の重心を時刻順にリストに格納します---")
	rest_list = []
	for row in zipped_list:
		if (row[4] == 0): #休息なら
			p, _, _ = extract_mean(row[1], row[2], row[3])
			rest_list.append(((row[0][1] - row[0][0]).total_seconds() / 60, p[0], p[1])) # (時間 [minutes], lat, lon)
	print("---休息時間と休息の重心を時刻順にリストに格納しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return rest_list

#休息でない行動のみのリストにする
"""
Parameter
	zipped_list	: 圧縮後のリスト
return
	walk_list	: 行動のリスト（他の要素はなにも変えない）
"""
def make_walk_data(zipped_list):
	print(sys._getframe().f_code.co_name, "実行中")
	print("休息以外の行動のみを取り出します---")
	walk_list = []
	for row in zipped_list:
		if (row[4] != 0):
			p, _, _ = extract_mean(row[1], row[2], row[3])
			walk_list.append(((row[0][1] - row[0][0]).total_seconds() / 60, p[0], p[1])) # (時間 [minutes], lat, lon)
	print("---休息以外の行動のみをリストに格納しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return walk_list

#歩行の変化点検出を行う
#どのように行うかは未定，歩行の前後にウォーミングアップのような時間が存在するため
"""
	t_list	:圧縮後の時刻のリスト
	p_list	:圧縮後の位置情報のリスト
	d_list	:圧縮後の距離のリスト
	l_list	:圧縮後の暫定的なラベルのリスト
"""
"""
def devide_walk_point(t_list, p_list, d_list, v_list, l_list):
"""

#特徴をCSVにして出力する (圧縮が既に行われている前提) 
"""
Parameter
	filename	:ファイルのパス
	t_list	:圧縮後の時刻のリスト
	p_list	:圧縮後の位置情報のリスト
	d_list	:圧縮後の距離のリスト
	l_list	:圧縮後の暫定的なラベルのリスト
"""
def output_feature_info(filename, t_list, p_list, d_list, v_list, l_list):
	print(sys._getframe().f_code.co_name, "実行中")

	###登録に必要な変数###
	before_lat = None
	before_lon = None
	after_lat = None
	after_lon = None

	#####登録情報#####
	time_index = None
	center = 0.0 # 移動の重心
	time_length = None #圧縮にまとめられた観測の個数
	moving_distance = None #休息間の距離
	moving_direction = None #次の休息への移動方向
	previous_rest_length = None #前の休息の長さ

	print("特徴を計算します---")
	feature_list =[]
	#####登録#####
	for i, (time, pos, dis, vel, label) in enumerate(zip(t_list, p_list, d_list, v_list, l_list)):
		if (label == 1): # 歩行
			time_index = time
			time_length = (time[1] - time[0]).total_seconds() / 5 + 1
			max_vel = max(vel)
			min_vel = min(vel)
			
			center, dis, mean_vel = extract_mean(pos, dis, vel)
			ave_accumulated_distance = dis # 行動内での移動距離の１観測あたり

		if (label == 0): # 休息
			###前後関係に着目した特徴の算出###
			after_lat = pos[0][0]
			after_lon = pos[0][1]
			if (before_lat is not None and before_lon is not None):
				moving_distance, moving_direction = geo.get_distance_and_direction(before_lat, before_lon, after_lat, after_lon, True) #前の重心との直線距離
				DA = calculate_rate(moving_distance, ave_accumulated_distance)
				AD = calculate_rate(ave_accumulated_distance, moving_distance)
				VA = 5 * calculate_rate(mean_vel, ave_accumulated_distance)
				
				###リストに追加###
				feature_list.append([time_index, center, time_length, ave_accumulated_distance, mean_vel, max_vel, min_vel, moving_distance, moving_direction, previous_rest_length, label, DA, AD, VA])

			###引継###
			previous_rest_length = (time[1] - time[0]).total_seconds() / 5 + 1
			before_lat = pos[len(pos) - 1][0]
			before_lon = pos[len(pos) - 1][1]
				
	print("---特徴を計算しました")

	print(filename + "に出力します---")
	#####出力#####
	with open(filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(("Time", "Place", "Continuous time", "Moving amount", "Average velocity", "Max velocity", "Min velocity", "Distance", "Moving direction", "Last rest length", "Label", "D/A", "A/D", "V/A"))
		for feature in feature_list:
			writer.writerow(feature)
	print("---" + filename + "に出力しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return

# 場所，距離，速さに関してリスト内のそれぞれ重心，総和，平均を求める
def extract_mean(p_list, d_list, v_list):
	ave_lat_list = [] #平均を求めるための緯度のリスト
	ave_lon_list = [] #平均を求めるための経度のリスト
	ave_dis_list = [] #休息中の距離の総和（おそらく休息中の移動距離の総和は0）を１観測あたりに直した距離のリスト
	ave_vel_list = [] #休息中の速さの平均を求めるための速さのリスト
	for place, distance, velocity in zip(p_list, d_list, v_list):
		ave_lat_list.append(place[0])
		ave_lon_list.append(place[1])
		ave_dis_list.append(distance)
		ave_vel_list.append(velocity)
	
	lat = sum(ave_lat_list) / len(ave_lat_list)
	lon = sum(ave_lon_list) / len(ave_lon_list)
	dis = sum(ave_dis_list) / len(ave_dis_list)
	vel = sum(ave_vel_list) / len(ave_vel_list)
	return (lat,lon), dis, vel

# 比を算出する
def calculate_rate(amount1, amount2):
	if (amount2 != 0):
		return amount1 / amount2
	else:
		return 0

#3次元のデータを主成分分析し，2次元にする
def reduce_dim_from3_to2(x, y, z):
    print("今から主成分分析を行います")
    features = np.array([x.values, y.values, z.values]).T
    pca = skd.PCA()
    pca.fit(features)
    transformed = pca.fit_transform(features)
    print("累積寄与率: ", pca.explained_variance_ratio_)
    print("主成分分析が終了しました")
    return transformed[:, 0], transformed[:, 1]

#移動速度に対して分類を行い視覚化を可能にする
"""
Parameter
	v_list	:速さのリスト
	l_list	:ラベルのリスト．Noneであれば速さだけで分類を行い，Noneでなければこのラベルを元に分類を行う
return
	それぞれが振り分けられたクラスタ ("red", "green", "blue")
"""
def calassify_velocity(v_list, l_list=None):
	data_list = []
	if (l_list is None):
		for data in v_list:
			if (choice_state(data) == 0):
				data_list.append("red")
			elif (choice_state(data) == 1):
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
	t_list	:圧縮前の時間のリスト（いつも5sの間隔で作られている訳ではないので元の時間のリストの時間を参照する）
Return
	l_list	:解凍したラベルのリスト
"""
def decode(t_list, zipped_t_list, zipped_l_list):
	print(sys._getframe().f_code.co_name, "実行中")
	index = 0
	l_list = []

	start = zipped_t_list[0][0]
	end = zipped_t_list[0][1]
	label = zipped_l_list[0]
	for time in t_list:
		if (start <= time and time <= end):
			l_list.append(label)
		if (len(l_list) == len(t_list)):
			break
		if (end <= time):
			index += 1
			start = zipped_t_list[index][0]
			end = zipped_t_list[index][1]
			label = zipped_l_list[index]

	print(sys._getframe().f_code.co_name, "正常終了\n")
	return l_list

def decode2(l_list, new_l_list):
	print(sys._getframe().f_code.co_name, "実行中")
	index = 0
	new_labels = []
	for l in l_list:
		if (l == 1):
			new_labels.append(new_l_list[index] + 1)
			index += 1
		else:
			new_labels.append(l)
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return new_labels

def temp(d_list, v_list):
	r_list = []
	for d, v in zip(d_list, v_list):
		if (v != 0):
			r_list.append(d - (5 * v))
		else:
			r_list.append(0)
	return r_list

if __name__ == '__main__':
	filename = "behavior_classification/features.csv"
	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
	time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(20158, start, end) #2次元リスト (1日分 * 日数分)
	for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
		# ---前処理---
		t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
		
		# 畳み込み
		v_list = preprocessing.convolution(v_list, 3)
		d_list = preprocessing.convolution(d_list, 3)
		t_list = preprocessing.elimination(t_list, 3)
		p_list = preprocessing.elimination(p_list, 3)
		a_list = preprocessing.elimination(a_list, 3)

		# 時系列描画
		#plot_velocity_data(t_list, v_list)
		c_list = calassify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
		r_list = temp(d_list, v_list)
		scatter_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示
		scatter_plot(t_list, d_list, c_list) #時系列で速さの散布図を表示
		scatter_plot(t_list, r_list, c_list) #時系列で速さの散布図を表示

		# 圧縮操作
		zipped_list = zipping(t_list, p_list, d_list, v_list) # 圧縮する

		#c_list = calassify_velocity([row[3] for row in zipped_list]) # クラスタ分けを行う (速さを3つに分類しているだけ)
		#scatter_plot([row[0] for row in zipped_list], [row[2] for row in zipped_list], c_list) # 時系列で速さの散布図を表示
		#scatter_plot([row[0] for row in zipped_list], [row[3] for row in zipped_list], c_list) # 時系列で速さの散布図を表示

		# ---特徴抽出---
		output_feature_info(filename, [row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
		
		# 各特徴
		df = pd.read_csv(filepath_or_buffer = filename, encoding = "utf-8", sep = ",", header = 0, usecols = [0,1,2,3,4,5,6,7,9,11,12,13], names=('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L', 'M', 'N'))
		
		# 軌跡描画
		display = disp.Adjectory(True)
		zipped_rest_list = make_rest_data(zipped_list) # 描画用に休息時間と重心だけのリストにする
		display.plot_rest_place(zipped_rest_list) # 休息の場所の分布のプロット
		display.plot_moving_ad(df['B'].tolist()) # 移動の軌跡をプロット
		plt.show()
		
		
		
		#result = decode2([row[4] for row in zipped_list], pred)
		
		# 時系列描画
		c_list = decode(t_list, [row[0] for row in zipped_list], result)
		scatter_plot(t_list, v_list, c_list) # 時系列で速さの散布図を表示

		
		#df = pd.read_csv(filepath_or_buffer = "features.csv", encoding = "utf-8", sep = ",", header = 0, usecols = [0,3,4,5,6,8,9,10], names=('A', 'D', 'E', 'F', 'G', 'I', 'J', 'K'))
		#b, c = reduce_dim_from3_to2(df['E'], df['F'], df['I'])
		observation = np.array(c_list).reshape(-1, 1)
		interface = hmm.hmm_interface(5)
		interface.train_data(observation)
		print("遷移行列: ",interface.transition_matrix)
		print("出力期待値: ",interface.means)
		print("初期確率: ",interface.init_matrix)
		result = interface.predict_data(observation)
		c_list = decode(t_list, [row[0] for row in zipped_list], result)
		scatter_plot(t_list, v_list, c_list) # 時系列で速さの散布図を表示
		