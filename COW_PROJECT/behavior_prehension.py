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

# 自作メソッド
import cows.geography as geo
import behavior_classification.functions.hmm as hmm
import behavior_classification.functions.loading as loading
import behavior_classification.functions.preprocessing as preprocessing
import behavior_classification.functions.plotting as plotting
import behavior_classification.functions.analyzing as analyzing
import behavior_classification.functions.regex as regex
import behavior_classification.functions.postprocessing as postprocessing
import behavior_classification.functions.output_features as output_features
import image.adjectory_image as disp

""" 基本的な手順はoutput_features.output_features()に集約されているが，個々のグラフなどの可視化にはこちらのメインを活用 """

if __name__ == '__main__':
	filename = "behavior_classification/training_data/features.csv"
	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
	print(os.getcwd())
	time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(20158, start, end) #2次元リスト (1日分 * 日数分)
	if (len(position_list[0]) != 0):
		for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
			# ---前処理---
			t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
			
			# 畳み込み
			#v_list = preprocessing.convolution(v_list, 3)
			#d_list = preprocessing.convolution(d_list, 3)
			#t_list = preprocessing.elimination(t_list, 3)
			#p_list = preprocessing.elimination(p_list, 3)
			#a_list = preprocessing.elimination(a_list, 3)

			# 時系列描画
			#plotting.line_plot(t_list, v_list)
			c_list = output_features.classify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
			plotting.scatter_time_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示
			#plotting.scatter_time_plot(t_list, d_list, c_list) #時系列で速さの散布図を表示

			# 圧縮操作
			zipped_list = output_features.compress(t_list, p_list, d_list, v_list) # 圧縮する

			#c_list = output_features.classify_velocity([row[3] for row in zipped_list]) # クラスタ分けを行う (速さを3つに分類しているだけ)
			#plotting.scatter_time_plot([row[0] for row in zipped_list], [row[2] for row in zipped_list], c_list) # 時系列で速さの散布図を表示
			#plotting.scatter_time_plot([row[0] for row in zipped_list], [row[3] for row in zipped_list], c_list) # 時系列で速さの散布図を表示

			# ---特徴抽出---
			output_features.output_feature_info(filename, [row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
			
			# 各特徴
			df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,1,2,3,4,5,6,7,9], names=('Time', 'RCategory', 'WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance')) # csv読み込み
			plotting.show_3d_plot(sp.stats.zscore(df['RTime']), sp.stats.zscore(df['WTime']), sp.stats.zscore(df['AccumulatedDis'])) # 3次元プロット
			x, y = analyzing.reduce_dim_from3_to2(sp.stats.zscore(df['RTime']), sp.stats.zscore(df['WTime']), sp.stats.zscore(df['AccumulatedDis'])) # 主成分分析 (3 → 2)
			plotting.time_scatter(df['Time'].tolist(), x, y) # 時系列プロット

			# 軌跡描画
			display = disp.Adjectory(True)
			zipped_rest_list = output_features.extract_one_behavior(zipped_list, state = "resting") # 描画用に休息時間と重心だけのリストにする
			display.plot_rest_place(zipped_rest_list) # 休息の場所の分布のプロット
			zipped_walk_list = np.array(output_features.extract_one_behavior(zipped_list, state = "walking")) # 描画用に歩行時間と重心だけのリストにする
			display.plot_moving_ad(zipped_walk_list[:,1:].tolist()) # 移動の軌跡をプロット
			plt.show()
			
			# --- 分析 ---
			filename1 = "behavior_classification/bst/model.pickle"
			filename2 = "behavior_classification/bst/model2.pickle"
		
			labels = []
			probs = []
			model1 = joblib.load(filename1)
			model2 = joblib.load(filename2)
			x1, x2, x3, x4, x5, x6, x7, x8 = df['RCategory'].tolist(), df['WCategory'].tolist(), df['RTime'].tolist(), df['WTime'].tolist(), df['AccumulatedDis'].tolist(), df['Velocity'].tolist(), df['MVelocity'].tolist(), df['Distance'].tolist()
			x = np.array((x1, x2, x3, x4, x5, x6, x7, x8)).T
			result1 = model1.predict(x)
			result2 = model2.predict(x)
			prob1 = model1.predict_proba(x)
			prob2 = model2.predict_proba(x)
			print(result1)
			print(result2)	
			for a, b, c, d in zip(result1, result2, prob1, prob2):
				labels.append(a)
				labels.append(b)
				probs.append(np.insert(c, 2, 0.0))
				probs.append(d)

			# --- 復元 ---
			zipped_t_list = regex.str_to_datetime(df['Time'].tolist())
			new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
			_, probs = postprocessing.decompress(t_list, zipped_t_list, probs)

			# --- 隠れマルコフモデルに当てはめる ---
			#interface = hmm.hmm_interface(3)
			#interface.train_data(probs)
			#labels = interface.predict_data(probs)

			new_v_list = postprocessing.make_new_list(t_list, new_t_list, v_list)
			plotting.scatter_time_plot(new_t_list, new_v_list, labels) # 時系列で速さの散布図を表示
			