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
import pdb

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
import behavior_classification.myClass.feature_extraction as feature_extraction
import behavior_classification.myClass.plotting as my_plot
import behavior_classification.myClass.evaluation as evaluation
import behavior_classification.models.beysian.single_model as single_model
import behavior_classification.models.beysian.mixed_model as mixed_model

def get_prior_dist(segment_name, behavior_name, usecols, names):
	""" 事前分布の設定のために教師データから知見を得る """
	rest_dataset_file = "behavior_classification/training_data/rest_train_data.csv" # 休息の訓練データ
	walk_dataset_file = "behavior_classification/training_data/walk_train_data.csv" # 歩行の訓練データ
	graze_dataset_file = "behavior_classification/training_data/graze_train_data.csv" # 採食の訓練データ
	# 教師データから分布を取得
	if (behavior_name is "rest"):
		rest_df = pd.read_csv(rest_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
		dist = single_model.MyGaussianDistribution(rest_df)
	elif (behavior_name is "walk"):
		walk_df = pd.read_csv(walk_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
		dist = single_model.MyGaussianDistribution(walk_df)
	else:
		if (segment_name is "rest"):
			graze_df = pd.read_csv(graze_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
		elif (segment_name is "act"):
			graze_df = pd.read_csv(graze_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
		dist = single_model.MyGaussianDistribution(graze_df)
	return dist

if __name__ == '__main__':
	# --- 変数定義 ---
	rest_usecols, rest_names = [3,5,6,9,10], ['RTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv']
	walk_usecols, walk_names = [4,5,6,11,12], ['WTime', 'AccumulatedDis', 'VelocityAve', 'WalkVelocityAve', 'WalkVelocityDiv']
	savefile = "behavior_classification/training_data/features.csv"
	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
	target_cow_id = 20158

	# 事前分布を正解データから得る
	rest_dist = get_prior_dist("rest", "rest", rest_usecols, rest_names) # 休息の分布
	walk_dist = get_prior_dist("act", "walk", walk_usecols, walk_names) # 歩行の分布
	graze_dist_r = get_prior_dist("rest", "graze", rest_usecols, rest_names) # 採食の分布
	graze_dist_a = get_prior_dist("act", "graze", walk_usecols, walk_names) # 採食の分布
	# 事前パラメータを用意
	cov_matrixes_r = [rest_dist.get_cov_matrix(), graze_dist_r.get_cov_matrix()]
	mu_vectors_r = [rest_dist.get_mean_vector(), graze_dist_r.get_mean_vector()]
	pi_vector_r = [0.4, 0.6]
	alpha_vector_r = [1, 1]

	cov_matrixes_w = [walk_dist.get_cov_matrix(), graze_dist_a.get_cov_matrix()]
	mu_vectors_w = [walk_dist.get_mean_vector(), graze_dist_a.get_mean_vector()]
	pi_vector_w = [0.3, 0.7]
	alpha_vector_w = [1, 1]
	max_iterater = 100

	# テスト用の1日のデータを読み込み
	time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(target_cow_id, start, end) #2次元リスト (1日分 * 日数分)
	if (len(position_list[0]) != 0):
		for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
			# --- 前処理 ---
			t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
			# --- 特徴抽出 ---
			features = feature_extraction.FeatureExtraction(savefile, start, target_cow_id)
			features.output_features()
			# --- 仮説検証 ---
			df = pd.read_csv(savefile, sep = ",", header = 0, usecols = [0,3,4,5,6,9,10,11,12], names=('Time', 'RTime', 'WTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv', 'WalkVelocityAve', 'WalkVelocityDiv')) # csv読み込み
			X_rest = df[rest_names].values.T
			X_walk = df[walk_names].values.T
			# ギブスサンプリングによるクラスタリング
			gaussian_model_rest = mixed_model.GaussianMixedModel(cov_matrixes_r, mu_vectors_r, pi_vector_r, alpha_vector_r, max_iterater)
			rest_result = gaussian_model_rest.gibbs_sample(X_rest, np.array([[0, 0, 0, 0, 0]]).T, 1, 6, np.eye(5))

			gaussian_model_walk = mixed_model.GaussianMixedModel(cov_matrixes_w, mu_vectors_w, pi_vector_w, alpha_vector_w, max_iterater)
			walk_result = gaussian_model_walk.gibbs_sample(X_walk, np.array([[0, 0, 0, 0, 0]]).T, 1, 6, np.eye(5))

			rest_result = postprocessing.process_result(rest_result)
			walk_result = postprocessing.process_result(walk_result)
			print(rest_result)
			print(walk_result)
			
			"""
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
	"""
