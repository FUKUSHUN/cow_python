#-*- encoding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import csv
import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib
import pickle

# 自作クラス
import cows.geography as geo
import behavior_classification.hmm as hmm
import behavior_classification.loading as loading
import behavior_classification.preprocessing as preprocessing
import behavior_classification.plotting as plotting
import behavior_classification.analyzing as analyzing
import behavior_classification.regex as regex
import behavior_classification.postprocessing as postprocessing
import image.adjectory_image as disp

"""
えげつないスパゲッティです
このコードでは行動分類を試みています．
データの読み込みはloading.py, データの前処理はpreprocessing.pyに関数を作成し，呼び出すことにしており，画像生成やプロットに関してはplotting.pyとadjectory_image.pyに任せています
メイン関数をこのファイルに置き，必要な処理は個々にこのファイル内で関数定義しています
"""

"""
圧縮操作を行う
Parameter
	t_list	: 圧縮されていない時間のリスト
	p_list	: 圧縮されていない(緯度, 経度)のリスト
	d_list	: 圧縮されていない距離のリスト
	v_list	: 圧縮されていない速度のリスト
return
	zipped_list	: 各要素が (0: (start, end), 1: 重心 (緯度, 経度), 2: 総距離, 3: 平均速度, 4: 暫定的なラベル) の形のリスト
"""
def compress(t_list, p_list, d_list, v_list):
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

"""
休息・採食・歩行を判断する（今は速度データをもとに閾値や最近傍のアプローチだが変更する可能性あり）
"""
def choice_state(velocity, r_threshold = 0.0694, g_threshold = 0.181):
	if (velocity < r_threshold):
		return 0 # 休息
	elif (r_threshold <= velocity and velocity < g_threshold):
		return 1 # 採食
	else:
		return 1 # 歩行 (実施不透明)

"""
ある行動のみを個別に取り出す
行動時間と行動の重心（緯度・経度）を時刻順に格納したリストを返す
restingであれば休息のみ，walkingであれば歩行のみをそれぞれ個別に取り出す
Parameter
	zipped_list		: 圧縮後のリスト (各要素の１番目が (start, end), 2番目が (緯度, 経度), 3番目が速度, 4番目にラベル)
	state	: 休息を取り出す場合には"resting", 歩行を取り出す場合には"walking"
return
	rest_list	: 休息の重心とその時間
"""
def extract_one_behavior(zipped_list, state="resting"):
	behavior_dict = {"resting":0, "walking":1} # 行動ラベルの辞書
	print(sys._getframe().f_code.co_name, "実行中")
	print(state + "時間と" + state +"重心を時刻順にリストに格納します---")
	behavior_list = []
	for row in zipped_list:
		if (row[4] == behavior_dict[state]): #休息or歩行なら
			p, _, _ = extract_mean(row[1], row[2], row[3])
			behavior_list.append([(row[0][1] - row[0][0]).total_seconds() / 60, p[0], p[1]]) # (時間 [minutes], lat, lon)
	print("---state" + "時間と" + state +"重心を時刻順にリストに格納しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return behavior_list

"""
歩行の変化点検出を行う
どのように行うかは未定，歩行の前後にウォーミングアップのような時間が存在するため
Parameter
	t_list	:圧縮後の時刻のリスト
	p_list	:圧縮後の位置情報のリスト
	d_list	:圧縮後の距離のリスト
	l_list	:圧縮後の暫定的なラベルのリスト
"""
"""
def devide_walk_point(t_list, p_list, d_list, v_list, l_list):
"""

"""
特徴をCSVにして出力する (圧縮が既に行われている前提) 
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
	center = None # 移動の重心
	previous_rest_length = None #圧縮にまとめられた前の休息の観測の個数
	walking_length = None #圧縮にまとめられた歩行の観測の個数
	moving_distance = None #休息間の距離
	moving_direction = None #次の休息への移動方向

	print("特徴を計算します---")
	feature_list =[]
	behavior_dict = {"resting":0, "walking":1} # 行動ラベルの辞書
	#####登録#####
	for i, (time, pos, dis, vel, label) in enumerate(zip(t_list, p_list, d_list, v_list, l_list)):
		if (label == behavior_dict["walking"]): # 歩行
			if (i != 0):
				time_index += time[0].strftime("%Y/%m/%d %H:%M:%S") + "-" + time[1].strftime("%Y/%m/%d %H:%M:%S")
				walking_length = (time[1] - time[0]).total_seconds() / 5 + 1
				max_vel = max(vel)
				min_vel = min(vel)
				
				center, dis, mean_vel = extract_mean(pos, dis, vel)
				center = str(center[0]) + "-" + str(center[1])
				ave_accumulated_distance = dis # 行動内での移動距離の１観測あたり

		if (label == behavior_dict["resting"]): # 休息
			###前後関係に着目した特徴の算出###
			after_lat = pos[0][0]
			after_lon = pos[0][1]
			if (before_lat is not None and before_lon is not None):
				moving_distance, moving_direction = geo.get_distance_and_direction(before_lat, before_lon, after_lat, after_lon, True) #前の重心との直線距離
				DA = calculate_rate(moving_distance, ave_accumulated_distance)
				AD = calculate_rate(ave_accumulated_distance, moving_distance)
				VA = 5 * calculate_rate(mean_vel, ave_accumulated_distance)
				
				###リストに追加###
				feature_list.append([time_index, center, previous_rest_length, walking_length, ave_accumulated_distance, mean_vel, max_vel, min_vel, moving_distance, moving_direction, DA, AD, VA])

			###引継###
			previous_rest_length = (time[1] - time[0]).total_seconds() / 5 + 1
			before_lat = pos[len(pos) - 1][0]
			before_lon = pos[len(pos) - 1][1]
			time_index = time[0].strftime("%Y/%m/%d %H:%M:%S") + "-" + time[1].strftime("%Y/%m/%d %H:%M:%S") + " | "
				
	print("---特徴を計算しました")

	print(filename + "に出力します---")
	#####出力#####
	with open(filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(("Time", "Walking place", "Last rest time", "Walking time", "Moving amount", "Average velocity", "Max velocity", "Min velocity", "Distance", "Direction", "D/A", "A/D", "V/A"))
		for feature in feature_list:
			writer.writerow(feature)
	print("---" + filename + "に出力しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return

"""
場所，距離，速さに関してリスト内のそれぞれ重心，総和，平均を求める
Parameter
	p_list	: (緯度，経度) のリスト
	d_list	: 距離のリスト
	v_list	: 速さのリスト
"""
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
"""
比を算出する
"""
def calculate_rate(amount1, amount2):
	if (amount2 != 0):
		return amount1 / amount2
	else:
		return 0

"""
移動速度に対して分類を行い視覚化を可能にする
Parameter
	v_list	:速さのリスト
return
	それぞれが振り分けられたクラスタ ("red", "green", "blue")
"""
def classify_velocity(v_list):
	data_list = []
	for data in v_list:
		if (choice_state(data) == 0):
			data_list.append("red")
		elif (choice_state(data) == 1):
			data_list.append("green")
		else:
			data_list.append("blue")
	return data_list

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
		#plotting.line_plot(t_list, v_list)
		c_list = classify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
		plotting.scatter_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示
		plotting.scatter_plot(t_list, d_list, c_list) #時系列で速さの散布図を表示

		# 圧縮操作
		zipped_list = compress(t_list, p_list, d_list, v_list) # 圧縮する

		#c_list = classify_velocity([row[3] for row in zipped_list]) # クラスタ分けを行う (速さを3つに分類しているだけ)
		#plotting.scatter_plot([row[0] for row in zipped_list], [row[2] for row in zipped_list], c_list) # 時系列で速さの散布図を表示
		#plotting.scatter_plot([row[0] for row in zipped_list], [row[3] for row in zipped_list], c_list) # 時系列で速さの散布図を表示

		# ---特徴抽出---
		output_feature_info(filename, [row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
		
		# 各特徴
		df = pd.read_csv(filepath_or_buffer = filename, encoding = "utf-8", sep = ",", header = 0, usecols = [0,1,2,3,4,5,6,7,8,10,11,12], names=('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M')) # csv読み込み
		plotting.show_3d_plot(sp.stats.zscore(df['C']), sp.stats.zscore(df['D']), sp.stats.zscore(df['I'])) # 3次元プロット
		x, y = analyzing.reduce_dim_from3_to2(sp.stats.zscore(df['C']), sp.stats.zscore(df['D']), sp.stats.zscore(df['I'])) # 主成分分析 (3 → 2)
		plotting.time_scatter(df['A'].tolist(), x, y) # 時系列プロット

		# 軌跡描画
		display = disp.Adjectory(True)
		zipped_rest_list = extract_one_behavior(zipped_list, state = "resting") # 描画用に休息時間と重心だけのリストにする
		display.plot_rest_place(zipped_rest_list) # 休息の場所の分布のプロット
		zipped_walk_list = np.array(extract_one_behavior(zipped_list, state = "walking")) # 描画用に歩行時間と重心だけのリストにする
		display.plot_moving_ad(zipped_walk_list[:,1:].tolist()) # 移動の軌跡をプロット
		plt.show()

		# --- 分析 ---
		filename1 = "behavior_classification/svm/model.pickle"
		filename2 = "behavior_classification/svm/model2.pickle"
	
		labels = []
		model1 = joblib.load(filename1)
		model2 = joblib.load(filename2)
		x1, x2, x3, x4, x5 = df['C'].tolist(), df['D'].tolist(), df['F'].tolist(), df['G'].tolist(), df['I'].tolist()
		x = np.array((x1, x2, x3, x4, x5)).T
		result1 = model1.predict(x)
		result2 = model2.predict(x)
		print(result1)
		print(result2)	
		for a, b in zip(result1, result2):
			#print(a, b)	
			#labels.append(np.argmax(a))
			#labels.append(np.argmax(b))
			labels.append(a)
			labels.append(b)

		"""
		observation = np.array([x,y]).T
		interface = hmm.hmm_interface(5)
		interface.train_data(observation)
		print("遷移行列: ",interface.transition_matrix)
		print("出力期待値: ",interface.means)
		print("初期確率: ",interface.init_matrix)
		result = interface.predict_data(observation)
		plotting.time_scatter(df['A'].tolist(), x, y, result)
		"""

		# --- 復元 ---
		zipped_t_list = regex.str_to_datetime(df['A'].tolist())
		#result = postprocessing.make_labels(result)
		new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
		new_v_list = postprocessing.make_new_list(t_list, new_t_list, v_list)
		#print(labels)
		plotting.scatter_plot(new_t_list, new_v_list, labels) # 時系列で速さの散布図を表示
		