#-*- encoding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import csv
import datetime
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib
import pickle

# 自作クラス
added_path = os.path.abspath('../')
sys.path.append(added_path)
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
このコードでは行動分類のための特徴出力までを行います．
データの読み込みはloading.py, データの前処理はpreprocessing.pyにある関数を呼び出すことにしており，画像生成やプロットに関してはplotting.pyとadjectory_image.pyに任せています
そのほかの必要な処理は個々にこのファイル内で関数定義しています (そこがスパゲッティ)
"""


def compress(t_list, p_list, d_list, v_list):
	""" 圧縮操作を行う
	Parameter
		t_list	: 圧縮されていない時間のリスト
		p_list	: 圧縮されていない(緯度, 経度)のリスト
		d_list	: 圧縮されていない距離のリスト
		v_list	: 圧縮されていない速度のリスト
	return
		zipped_list	: 各要素が (0: (start, end), 1: 重心 (緯度, 経度), 2: 総距離, 3: 平均速度, 4: 暫定的なラベル) の形のリスト """
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


def choice_state(velocity, r_threshold = 0.0694, g_threshold = 0.181):
	""" 休息・採食・歩行を判断する（今は速度データをもとに閾値や最近傍のアプローチだが変更する可能性あり）"""
	if (velocity < r_threshold):
		return 0 # 休息
	elif (r_threshold <= velocity and velocity < g_threshold):
		return 1 # 採食
	else:
		return 1 # 歩行 (実施不透明)


def extract_one_behavior(zipped_list, state="resting"):
	""" ある行動のみを個別に取り出す
	行動時間と行動の重心（緯度・経度）を時刻順に格納したリストを返す
	restingであれば休息のみ，walkingであれば歩行のみをそれぞれ個別に取り出す
	Parameter
		zipped_list		: 圧縮後のリスト (各要素の１番目が (start, end), 2番目が (緯度, 経度), 3番目が速度, 4番目にラベル)
		state	: 休息を取り出す場合には"resting", 歩行を取り出す場合には"walking"
	return
		rest_list	: 休息の重心とその時間 """
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


def output_feature_info(filename, t_list, p_list, d_list, v_list, l_list):
	""" 特徴をCSVにして出力する (圧縮が既に行われている前提) 
	Parameter
		filename	:ファイルのパス
		t_list	:圧縮後の時刻のリスト
		p_list	:圧縮後の位置情報のリスト
		d_list	:圧縮後の距離のリスト
		l_list	:圧縮後の暫定的なラベルのリスト """
	print(sys._getframe().f_code.co_name, "実行中")

	###登録に必要な変数###
	before_lat = None
	before_lon = None
	after_lat = None
	after_lon = None

	#####登録情報#####
	time_index = None
	resting_time_category = None # 時間帯のカテゴリ (日の出・日の入時刻を元に算出)
	walking_time_category = None # 時間帯のカテゴリ (日の出・日の入時刻を元に算出)
	previous_rest_length = None #圧縮にまとめられた前の休息の観測の個数
	walking_length = None #圧縮にまとめられた歩行の観測の個数
	moving_distance = None #休息間の距離
	moving_direction = None #次の休息への移動方向

	print("特徴を計算します---")
	feature_list =[]
	behavior_dict = {"resting":0, "walking":1} # 行動ラベルの辞書
	initial_datetime = t_list[0][0]
	#####登録#####
	for i, (time, pos, dis, vel, label) in enumerate(zip(t_list, p_list, d_list, v_list, l_list)):
		if (label == behavior_dict["walking"]): # 歩行
			if (i != 0): # 最初は休息から始まるようにする (もし最初が歩行ならそのデータは削られる)
				time_index += time[0].strftime("%Y/%m/%d %H:%M:%S") + "-" + time[1].strftime("%Y/%m/%d %H:%M:%S")
				walking_time_category = decide_time_category(time[0], initial_datetime)
				walking_length = (time[1] - time[0]).total_seconds() / 5 + 1
				max_vel = max(vel)
				min_vel = min(vel)
				
				_, dis, mean_vel = extract_mean(pos, dis, vel)
				ave_accumulated_distance = dis # 行動内での移動距離の１観測あたり

		if (label == behavior_dict["resting"]): # 休息
			###前後関係に着目した特徴の算出###
			after_lat = pos[0][0]
			after_lon = pos[0][1]
			resting_time_category = decide_time_category(time[0], initial_datetime)
			if (before_lat is not None and before_lon is not None):
				moving_distance, moving_direction = geo.get_distance_and_direction(before_lat, before_lon, after_lat, after_lon, True) #前の重心との直線距離
				
				###リストに追加###
				feature_list.append([time_index, resting_time_category, walking_time_category, previous_rest_length, walking_length, ave_accumulated_distance, mean_vel, max_vel, min_vel, moving_distance, moving_direction])

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
		writer.writerow(("Time", "Resting time category", "Walking time category", "Last rest time", "Walking time", "Moving amount", "Average velocity", "Max velocity", "Min velocity", "Distance", "Direction"))
		for feature in feature_list:
			writer.writerow(feature)
	print("---" + filename + "に出力しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return


def decide_time_category(dt, date):
	""" 時間帯に応じてカテゴリ変数を作成する """
	sunrise = datetime.datetime(date.year, date.month, date.day, 7, 8, 16)
	sunset = datetime.datetime(date.year, date.month, date.day, 16, 59, 38)
	if (sunrise + datetime.timedelta(days = 1) <= dt):
		sunrise += datetime.timedelta(days = 1)
		sunset += datetime.timedelta(days = 1)
	elif (dt < sunrise):
		sunrise -= datetime.timedelta(days = 1)
		sunset -= datetime.timedelta(days = 1)
	day_length = (sunset - sunrise).total_seconds()
	return (sunset - dt).total_seconds() / day_length


def extract_mean(p_list, d_list, v_list):
	""" 場所，距離，速さに関してリスト内のそれぞれ重心，総和，平均を求める
	Parameter
		p_list	: (緯度，経度) のリスト
		d_list	: 距離のリスト
		v_list	: 速さのリスト """
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


def calculate_rate(amount1, amount2):
	""" 比を算出する """
	if (amount2 != 0):
		return amount1 / amount2
	else:
		return 0


def classify_velocity(v_list):
	""" 移動速度に対して分類を行い視覚化を可能にする
	Parameter
		v_list	:速さのリスト
	return
		それぞれが振り分けられたクラスタ ("red", "green", "blue") """
	data_list = []
	for data in v_list:
		if (choice_state(data) == 0):
			data_list.append("red")
		elif (choice_state(data) == 1):
			data_list.append("green")
		else:
			data_list.append("blue")
	return data_list

def output_features(filename, date:datetime, cow_id):
	""" 日付と牛の個体番号からその日のその牛の位置情報を用いて特徴のファイル出力を行う
	Parameters
		filename	: 保存するファイルの絶対パス
		date	: 日付	: datetime
		cow_id	: 牛の個体番号．この牛の特徴を出力する """	
	start = date
	end = date + datetime.timedelta(days=1)
	time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(cow_id, start, end) #2次元リスト (1日分 * 日数分)だが1日ずつの指定のため要素数は1
	t_list, p_list, d_list, v_list, a_list = time_list[0], position_list[0], distance_list[0], velocity_list[0], angle_list[0]
	if (len(p_list) != 0): # データがない場合は飛ばす
		# ---前処理---
		t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
		
		# 畳み込み
		#v_list = preprocessing.convolution(v_list, 3)
		#d_list = preprocessing.convolution(d_list, 3)
		#t_list = preprocessing.elimination(t_list, 3)
		#p_list = preprocessing.elimination(p_list, 3)
		#a_list = preprocessing.elimination(a_list, 3)

		# 圧縮操作
		zipped_list = compress(t_list, p_list, d_list, v_list) # 圧縮する

		# ---特徴抽出---
		output_feature_info(filename, [row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
		return True # 特徴出力に成功したのでTrueを返す
	else:
		return False # データがなければFalseを返す