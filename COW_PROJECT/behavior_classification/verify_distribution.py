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
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import behavior_classification.functions.loading as loading
import behavior_classification.functions.preprocessing as preprocessing
import behavior_classification.functions.plotting as plotting
import behavior_classification.functions.analyzing as analyzing
import behavior_classification.functions.regex as regex
import behavior_classification.functions.postprocessing as postprocessing
import behavior_classification.functions.output_features as output_features

#自作クラス
import behavior_classification.myClass.gaussian_distribution as my_gauss

if __name__ == '__main__':
	# --- 変数定義 ---
	filename = "behavior_classification/training_data/features.csv"
	rest_dataset_file = "behavior_classification/training_data/rest_train_data.csv"
	start = datetime.datetime(2019, 3, 20, 0, 0, 0)
	end = datetime.datetime(2019, 3, 21, 0, 0, 0)
	target_cow_id = 20128

	# 休息教師データから分布を取得
	rest_df = pd.read_csv(rest_dataset_file, sep = ",", header = 0, usecols = [3,5,6], names=('RTime', 'AccumulatedDis', 'VelocityAve')) # csv読み込み
	rest_dist = my_gauss.MyGaussianDistribution(rest_df)

	# テスト用の1日のデータを読み込み
	time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(target_cow_id, start, end) #2次元リスト (1日分 * 日数分)
	if (len(position_list[0]) != 0):
		for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
			# ---前処理---
			t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
			# ---時系列描画---
			c_list = output_features.classify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
			plotting.scatter_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示
			# --- セグメント化 ---
			zipped_list = output_features.compress(t_list, p_list, d_list, v_list) # 圧縮する
			# ---特徴抽出---
			output_features.output_feature_info(filename, [row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
			# --- 仮説検証 ---
			df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,3,5,6], names=('Time', 'RTime', 'AccumulatedDis', 'VelocityAve')) # csv読み込み
			dists = []
			labels = []
			for i, row in df.iterrows():
				x = np.array([[row[1], row[2], row[3]]]).T
				dis = rest_dist.get_mahalanobis_distance(x)
				dists.append(dis)
				if (dis <= 1):
					labels.append(2)
					labels.append(0)
				elif (dis <= 2):
					labels.append(1)
					labels.append(0)
				else:
					labels.append(0)
					labels.append(0)

			# --- 復元 ---
			zipped_t_list = regex.str_to_datetime(df['Time'].tolist())
			new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
			new_v_list = postprocessing.make_new_list(t_list, new_t_list, v_list)
			plotting.scatter_plot(new_t_list, new_v_list, labels) # 時系列で速さの散布図を表示
	
	# こっちが正解の横臥リスト
	correct_filename = "behavior_classification/validation_data/20158.csv"
	correct_df = pd.read_csv(correct_filename, sep = ",", header = 0, usecols = [0,3,4], names=('Time', 'Velocity', 'Label')) # csv読み込み
	plotting.scatter_plot(correct_df['Time'], correct_df['Velocity'], correct_df['Label'])