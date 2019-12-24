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
import behavior_classification.myClass.feature_extraction as feature_extraction
import behavior_classification.myClass.plotting as my_plot
import behavior_classification.myClass.evaluation as evaluation

if __name__ == '__main__':
	# --- 変数定義 ---
	savefile = "behavior_classification/training_data/features.csv"
	rest_dataset_file = "behavior_classification/training_data/rest_train_data.csv"
	walk_dataset_file = "behavior_classification/training_data/walk_train_data.csv"
	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
	target_cow_id = 20158

	# 教師データから分布を取得
	rest_df = pd.read_csv(rest_dataset_file, sep = ",", header = 0, usecols = [3,5,6,9,10], names=('RTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv')) # csv読み込み
	rest_dist = my_gauss.MyGaussianDistribution(rest_df)

	walk_df = pd.read_csv(walk_dataset_file, sep = ",", header = 0, usecols = [4,5,6,11,12], names=('WTime', 'AccumulatedDis', 'VelocityAve', 'WalkVelocityAve', 'WalkVelocityDiv')) # csv読み込み
	walk_dist = my_gauss.MyGaussianDistribution(walk_df)

	# テスト用の1日のデータを読み込み
	time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(target_cow_id, start, end) #2次元リスト (1日分 * 日数分)
	if (len(position_list[0]) != 0):
		for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
			# ---前処理---
			t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
			# ---時系列描画---
			c_list = output_features.classify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
			#plotting.scatter_time_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示
			# --- 特徴抽出 ---
			features = feature_extraction.FeatureExtraction(savefile, start, target_cow_id)
			features.output_features()
			# --- 仮説検証 ---
			df = pd.read_csv(savefile, sep = ",", header = 0, usecols = [0,3,4,5,6,9,10,11,12], names=('Time', 'RTime', 'WTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv', 'WalkVelocityAve', 'WalkVelocityDiv')) # csv読み込み
			dists = []
			labels = []
			for i, row in df.iterrows():
				x1 = np.array([[row[1], row[2], row[4], row[5], row[6]]]).T
				rest_dis = rest_dist.get_mahalanobis_distance(x1)
				dists.append(rest_dis)
				x2 = np.array([[row[1], row[3], row[4], row[7], row[8]]]).T
				walk_dis = walk_dist.get_mahalanobis_distance(x2)
				dists.append((rest_dis, walk_dis))
				if (rest_dis <= 2):
					labels.append(1)
				else:
					labels.append(0)
				if (walk_dis <= 2):
					labels.append(2)
				else:
					labels.append(0)

			# --- 復元 ---
			zipped_t_list = regex.str_to_datetime(df['Time'].tolist())
			new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
			new_v_list = postprocessing.make_new_list(t_list, new_t_list, v_list)
			pred_plot = my_plot.PlotUtility()
			pred_plot.scatter_time_plot(new_t_list, new_v_list, labels) # 時系列で速さの散布図を表示
	
	# こっちが正解の横臥リスト
	correct_filename = "behavior_classification/validation_data/20181230_20158.csv"
	correct_df = pd.read_csv(correct_filename, sep = ",", header = 0, usecols = [0,3,4], names=('Time', 'Velocity', 'Label')) # csv読み込み
	correct_plot = my_plot.PlotUtility()
	correct_plot.scatter_time_plot(correct_df['Time'], correct_df['Velocity'], correct_df['Label'])
	
	# 真偽，陰陽で評価を行う
	pred_plot.show()
	correct_plot.show()
	answers = correct_df['Label'].tolist()[1:]
	evaluater = evaluation.Evaluation(labels, answers)
	ev_rest = evaluater.evaluate(1)
	ev_walk = evaluater.evaluate(2)
	print(ev_rest)
	print(ev_walk)
