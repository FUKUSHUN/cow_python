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
import sklearn.decomposition as skd # 主成分分析
import pickle
import pdb # デバッグ用

# 自作メソッド
import behavior_classification.functions.loading as loading
import behavior_classification.functions.preprocessing as preprocessing
import behavior_classification.functions.plotting as plotting
import behavior_classification.functions.postprocessing as postprocessing

#自作クラス
import behavior_classification.myClass.feature_extraction as feature_extraction
import behavior_classification.myClass.plotting as my_plot
import behavior_classification.myClass.evaluation as evaluation
import behavior_classification.models.beysian.single_model as single_model
import behavior_classification.models.beysian.mixed_model as mixed_model

class Classifier:
	rest_usecols, rest_names = [3,5,9], ['RTime', 'AccumulatedDis', 'RestVelocityAve']
	walk_usecols, walk_names = [4,5,11], ['WTime', 'AccumulatedDis', 'WalkVelocityAve']
	rest_dist: single_model.MyGaussianDistribution # 休息の分布
	walk_dist: single_model.MyGaussianDistribution # 歩行の分布
	graze_dist_r: single_model.MyGaussianDistribution # 採食の分布
	graze_dist_a: single_model.MyGaussianDistribution # 採食の分布
	mixture_model_r: mixed_model.GaussianMixedModel # 休息セグメントの混合モデル
	mixture_model_a: mixed_model.GaussianMixedModel # 活動セグメントの混合モデル

	def __init__(self):
		self.rest_dist = self._get_prior_dist("rest", "rest", self.rest_usecols, self.rest_names) # 休息の分布
		self.walk_dist = self._get_prior_dist("act", "walk", self.walk_usecols, self.walk_names) # 歩行の分布
		self.graze_dist_r = self._get_prior_dist("rest", "graze", self.rest_usecols, self.rest_names) # 採食の分布
		self.graze_dist_a = self._get_prior_dist("act", "graze", self.walk_usecols, self.walk_names) # 採食の分布

	def _get_prior_dist(self, segment_name, behavior_name, usecols, names):
		""" 事前分布の設定のために教師データから知見を得る """
		rest_dataset_file = "behavior_classification/training_data/rest_train_data.csv" # 休息の訓練データ
		walk_dataset_file = "behavior_classification/training_data/walk_train_data.csv" # 歩行の訓練データ
		graze_dataset_file = "behavior_classification/training_data/graze_train_data.csv" # 採食の訓練データ
		# 教師データから分布を取得
		if (behavior_name == "rest"):
			rest_df = pd.read_csv(rest_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
			dist = single_model.MyGaussianDistribution(rest_df)
		elif (behavior_name == "walk"):
			walk_df = pd.read_csv(walk_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
			dist = single_model.MyGaussianDistribution(walk_df)
		else:
			if (segment_name == "rest"):
				graze_df = pd.read_csv(graze_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
			elif (segment_name == "act"):
				graze_df = pd.read_csv(graze_dataset_file, sep = ",", header = 0, usecols = usecols, names=names) # csv読み込み
			dist = single_model.MyGaussianDistribution(graze_df)
		return dist

	def visualize_pc(self, X, y, model, filename="output.gif"):
		""" 主成分を可視化し分布の形を得る """
		# print("今から主成分分析を行います")
		# pca = skd.PCA()
		# transformed = pca.fit_transform(X.T)
		# print("累積寄与率: ", pca.explained_variance_ratio_)
		# print("主成分分析が終了しました")
		# pl = my_plot.PlotUtility3D()
		# pl.plot_scatter(X[0,:], X[1,:], X[2,:], c=y)
		pl = my_plot.PlotUtility()
		pl.scatter_plot(X[0,:], X[1,:], y)
		# pl.save_fig(filename)
		pl.show()

		# 新たな入力に対する確率を推定
		new_X = np.arange(0, 100, 2)
		new_Y = np.arange(0, 1.00, 0.02)
		grid_X, grid_Y = np.meshgrid(new_X, new_Y)
		new_X = np.array([grid_X.ravel(), grid_Y.ravel()])
		prob_matrix = model.predict(new_X)
		plotter_prob = my_plot.PlotUtility3D()
		prob1, prob2 = prob_matrix[0], prob_matrix[1]
		plotter_prob.plot_surface(grid_X, grid_Y, prob1.reshape([50, 50]), c=1)
		plotter_prob.plot_surface(grid_X, grid_Y, prob2.reshape([50, 50]), c=2)
		pdb.set_trace()
		return

	def fit(self, date_list, cow_id_lists):
		""" 事前分布をもとに学習を行う (ある範囲の中でランダムにデータを取り出しfeature_listを作成し学習に使用)
			date_list:	list[datetime.datetime]	推論に使用する日付
			cow_id_lists:	list[str]	推論に使用する日付における牛の個体番号リスト """
		# --- 事前パラメータを用意 ---
		cov_matrixes_r = [self.rest_dist.get_cov_matrix(), self.graze_dist_r.get_cov_matrix()]
		mu_vectors_r = [self.rest_dist.get_mean_vector(), self.graze_dist_r.get_mean_vector()]
		pi_vector_r = [0.2, 0.8]
		alpha_vector_r = [1, 1]

		cov_matrixes_w = [self.graze_dist_a.get_cov_matrix(), self.walk_dist.get_cov_matrix()]
		mu_vectors_w = [self.graze_dist_a.get_mean_vector(), self.walk_dist.get_mean_vector()]
		pi_vector_w = [0.9, 0.1]
		alpha_vector_w = [1, 1]
		max_iterater = 50

		# --- 推論用のデータを生成 ---
		print("推論用のデータを生成します")
		feature_data = []
		for date, cow_id_list in zip(date_list, cow_id_lists):
			for cow_id in cow_id_list:
				# --- 特徴抽出 ---
				print("date: %s, cow_id: %s" %(date.strftime("%Y/%m/%d"), cow_id))
				features = feature_extraction.FeatureExtraction(date, cow_id)
				feature_list = features.output_features()
				if (len(feature_list) != 0):
					feature_data.extend(feature_list) # dataを追加
		np.random.shuffle(feature_data) # データをシャッフル
		feature_data = feature_data[: int(0.1 * len(feature_data))] # 無作為に抽出
		print("推論用のデータの生成が終わりました")

		# --- データをもとに学習を開始する ---
		print("パラメータの推論を行います")
		df = pd.DataFrame(data=feature_data, columns=["Time", "Resting time category", "Walking time category", "RTime", "WTime", "AccumulatedDis", "VelocityAve", "MaxVelocity", "MinVelocity", "RestVelocityAve", "RestVelocityDiv", "WalkVelocityAve", "WalkVelocityDiv", "Distance", "Direction"])
		X_rest = df[self.rest_names].values
		X_walk = df[self.walk_names].values
		# ギブスサンプリングによるクラスタリング (内部のクラスで事後分布のパラメータが記録されている)
		self.mixture_model_r = mixed_model.GaussianMixedModel(cov_matrixes_r, mu_vectors_r, pi_vector_r, alpha_vector_r, max_iterater)
		self.mixture_model_r.gibbs_sample(X_rest, np.array([[0, 0, 0]]).T, 1, 4, np.eye(3)) # 休息セグメントのクラスタリング
		self.mixture_model_a = mixed_model.GaussianMixedModel(cov_matrixes_w, mu_vectors_w, pi_vector_w, alpha_vector_w, max_iterater)
		self.mixture_model_a.gibbs_sample(X_walk, np.array([[0, 0, 0]]).T, 1, 4, np.eye(3)) # 歩行セグメントのクラスタリング
		print("パラメータの推論が終了しました")

		# --- パラメータを記録する ---
		mu_r, cov_r = self.mixture_model_r.get_gaussian_parameters()
		pi_r = self.mixture_model_r.get_pi_vector()
		mu_a, cov_a = self.mixture_model_a.get_gaussian_parameters()
		pi_a = self.mixture_model_a.get_pi_vector()
		filepath = "./behavior_information/parameter_log/" + date_list[0].strftime("%Y%m") + ".csv"
		self._write_parameter_log(mu_r, cov_r, pi_r, mu_a, cov_a, pi_a, filepath) # 事後分布のパラメータのログを残す
		return

	def classify(self, date, target_cow_id):
		""" 予測分布を元に行動を分類する
			date:	datetime.datetime
			target_cow_id:	str """
		t_list, p_list, d_list, v_list, a_list = loading.load_gps(target_cow_id, date) #2次元リスト (1日分)
		t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list, date) #牛舎内にいる時間を除く
		# --- 特徴抽出 ---
		features = feature_extraction.FeatureExtraction(date, target_cow_id)
		feature_list = features.output_features()
		if (len(feature_list) != 0):
			# --- 入力する特徴量の抽出 ---
			df = pd.DataFrame(data=feature_list, columns=["Time", "Resting time category", "Walking time category", "RTime", "WTime", "AccumulatedDis", "VelocityAve", "MaxVelocity", "MinVelocity", "RestVelocityAve", "RestVelocityDiv", "WalkVelocityAve", "WalkVelocityDiv", "Distance", "Direction"])
			df.to_csv("./behavior_classification/training_data/features.csv") # デバッグ用に残しておく（正常終了すれば次にこのメソッドが呼び出されたときに上書きされる）
			X_rest = df[self.rest_names].values
			X_walk = df[self.walk_names].values
			# --- 予測分布による分類 ---
			print("date: %s, cow_id: %s" %(date.strftime("%Y/%m/%d"), target_cow_id))
			rest_prob, _ = self.mixture_model_r.predict(X_rest)
			rest_result = np.argmax(rest_prob, axis=1)
			walk_prob, _ = self.mixture_model_a.predict(X_walk)
			walk_result = np.argmax(walk_prob, axis=1)
			walk_result = np.array([k+1 for k in walk_result]) # graze 0->1, walk 1->2
			df = pd.concat([df, pd.Series(data=rest_result, name='rest_prediction'), pd.Series(data=walk_result, name='walk_prediction')], axis=1)
			# --- 復元 ---
			zipped_t_list = []
			for time_index in df['Time'].tolist():
				zipped_t_list.append(time_index[0])
				zipped_t_list.append(time_index[1])
			labels = []
			for r, w in zip(rest_result, walk_result):
				labels.append(r)
				labels.append(w)
			new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
			new_v_list = postprocessing.make_new_list(t_list, new_t_list, v_list)
			# 12時から翌9時になるようにはみ出た部分をカットする
			ret_t, ret_v, ret_l = [], [], []
			for i, time in enumerate(new_t_list):
				if (12 <= time.hour or time.hour < 9):
					ret_t.append(new_t_list[i])
					ret_v.append(new_v_list[i])
					ret_l.append(labels[i])
				i += 1
			return ret_t, ret_v, ret_l
		else:
			return [], [], []
	
	def to_csv(self, t_list, v_list, labels, output_file):
		""" 行動レコードのCSVファイルを作成する """
		df = pd.concat([pd.Series(data=t_list, name='Time'), pd.Series(data=v_list, name='Velocity'), pd.Series(data=labels, name='Label')], axis=1)
		df.to_csv(output_file)
		return

	def plot_v_label(self, t_list, v_list, labels, output_file):
		""" 時系列の速さのリストを分類結果に応じて色分けする """
		pred_plot = my_plot.PlotUtility()
		pred_plot.scatter_time_plot(t_list, v_list, labels) # 時系列で速さの散布図を表示
		pred_plot.save_fig(output_file)
		return

	def _write_parameter_log(self, mu_r, cov_r, pi_r, mu_a, cov_a, pi_a, filepath):
		""" パラメータのログを取る """
		with open(filepath, mode='w', newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["pi (rest, graze)", pi_r.tolist()])
			writer.writerow(["average (rest)", mu_r[0].tolist(), "average (rest-graze)", mu_r[1].tolist()])
			writer.writerow(["cov (rest)", cov_r[0].tolist(), "cov (rest-graze)", cov_r[1].tolist()])
			writer.writerow(["pi (graze, walk)", pi_a.tolist()])
			writer.writerow(["average (act-graze)", mu_a[0].tolist(), "average (walk)", mu_a[1].tolist()])
			writer.writerow(["cov (act-graze)", cov_a[0].tolist(), "cov (walk)", cov_a[1].tolist()])
			writer.writerow(["\n"])
		print("%sにパラメータのログを出力しました" %filepath)
		return

# if __name__ == '__main__':
# 	# --- 変数定義 ---
# 	rest_usecols, rest_names = [3,5,6,9,10], ['RTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv']
# 	walk_usecols, walk_names = [4,5,6,11,12], ['WTime', 'AccumulatedDis', 'VelocityAve', 'WalkVelocityAve', 'WalkVelocityDiv']
# 	features_file = "behavior_classification/training_data/features.csv"
# 	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
# 	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
# 	target_cow_id = 20158

# 	# 事前分布を正解データから得る
# 	rest_dist = get_prior_dist("rest", "rest", rest_usecols, rest_names) # 休息の分布
# 	walk_dist = get_prior_dist("act", "walk", walk_usecols, walk_names) # 歩行の分布
# 	graze_dist_r = get_prior_dist("rest", "graze", rest_usecols, rest_names) # 採食の分布
# 	graze_dist_a = get_prior_dist("act", "graze", walk_usecols, walk_names) # 採食の分布
# 	# 事前パラメータを用意
# 	cov_matrixes_r = [rest_dist.get_cov_matrix(), graze_dist_r.get_cov_matrix()]
# 	mu_vectors_r = [rest_dist.get_mean_vector(), graze_dist_r.get_mean_vector()]
# 	pi_vector_r = [0.3, 0.7]
# 	alpha_vector_r = [1, 1]

# 	cov_matrixes_w = [graze_dist_a.get_cov_matrix(), walk_dist.get_cov_matrix()]
# 	mu_vectors_w = [graze_dist_a.get_mean_vector(), walk_dist.get_mean_vector()]
# 	pi_vector_w = [0.3, 0.7]
# 	alpha_vector_w = [1, 1]
# 	max_iterater = 50

# 	# テスト用の1日のデータを読み込み
# 	dt = start
# 	t_list, p_list, d_list, v_list, a_list = loading.load_gps(target_cow_id, dt) #2次元リスト (1日分 * 日数分)
# 	if (len(p_list) != 0):
# 		# --- 前処理 ---
# 		t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #牛舎内にいる時間を除く
# 		# --- 特徴抽出 ---
# 		features = feature_extraction.FeatureExtraction(features_file, dt, target_cow_id)
# 		features.output_features()
# 		# --- 仮説検証 ---
# 		df = pd.read_csv(features_file, sep = ",", header = 0, usecols = [0,3,4,5,6,9,10,11,12], names=('Time', 'RTime', 'WTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv', 'WalkVelocityAve', 'WalkVelocityDiv')) # csv読み込み
# 		X_rest = df[rest_names].values.T
# 		X_walk = df[walk_names].values.T
# 		# ギブスサンプリングによるクラスタリング
# 		gaussian_model_rest = mixed_model.GaussianMixedModel(cov_matrixes_r, mu_vectors_r, pi_vector_r, alpha_vector_r, max_iterater)
# 		rest_result = gaussian_model_rest.gibbs_sample(X_rest, np.array([[0, 0, 0, 0, 0]]).T, 1, 6, np.eye(5)) # 休息セグメントのクラスタリング
# 		gaussian_model_walk = mixed_model.GaussianMixedModel(cov_matrixes_w, mu_vectors_w, pi_vector_w, alpha_vector_w, max_iterater)
# 		walk_result = gaussian_model_walk.gibbs_sample(X_walk, np.array([[0, 0, 0, 0, 0]]).T, 1, 6, np.eye(5)) # 歩行セグメントのクラスタリング
# 		rest_result = postprocessing.process_result(rest_result, 0)
# 		walk_result = postprocessing.process_result(walk_result, 1)
# 		df = pd.concat([df, pd.Series(data=rest_result, name='rest_prediction'), pd.Series(data=walk_result, name='walk_prediction')], axis=1)
# 		df.to_csv("behavior_classification/prediction.csv")

# 		# 主成分分析
# 		#visualize_pc(X_rest, rest_result, gaussian_model_rest, "rest3D.gif")
# 		#visualize_pc(X_walk, walk_result, gaussian_model_walk, "walk3D.gif")
		
# 		# --- 復元 ---
# 		zipped_t_list = regex.str_to_datetime(df['Time'].tolist())
# 		labels = []
# 		for r, w in zip(rest_result, walk_result):
# 			labels.append(r)
# 			labels.append(w)
# 		new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
# 		new_v_list = postprocessing.make_new_list(t_list, new_t_list, v_list)
# 		pred_plot = my_plot.PlotUtility()
# 		pred_plot.scatter_time_plot(new_t_list, new_v_list, labels) # 時系列で速さの散布図を表示
	 
	
# 	# こっちが正解の横臥リスト
# 	correct_filename = "behavior_classification/validation_data/20181230_20158.csv"
# 	correct_df = pd.read_csv(correct_filename, sep = ",", header = 0, usecols = [0,3,4], names=('Time', 'Velocity', 'Label')) # csv読み込み
# 	correct_plot = my_plot.PlotUtility()
# 	correct_plot.scatter_time_plot(correct_df['Time'], correct_df['Velocity'], correct_df['Label'])
	
# 	# 真偽，陰陽で評価を行う
# 	pred_plot.show()
# 	# pred_plot.save_fig("prediction.png")
# 	correct_plot.show()
# 	# correct_plot.save_fig("correct.png")
# 	answers = correct_df['Label'].tolist()[1:]
# 	evaluater = evaluation.Evaluation(labels, answers)
# 	ev_rest = evaluater.evaluate(1)
# 	ev_walk = evaluater.evaluate(2)
# 	print(ev_rest)
# 	print(ev_walk)