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
import behavior_classification.functions.plotting as plotting

#自作クラス
import behavior_classification.myClass.gaussian_distribution as my_gauss
import behavior_classification.myClass.plotting as my_plot

if __name__ == '__main__':
    # --- 変数定義 ---
    rest_dataset_file = "behavior_classification/training_data/rest_train_data.csv"
    walk_dataset_file = "behavior_classification/training_data/walk_train_data.csv"
    savefile = "behavior_classification/training_data/features.csv"

    # 教師データから分布を取得
    rest_df = pd.read_csv(rest_dataset_file, sep = ",", header = 0, usecols = [3,5,6,9,10], names=('RTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv')) # csv読み込み
    rest_dist = my_gauss.MyGaussianDistribution(rest_df)

    walk_df = pd.read_csv(walk_dataset_file, sep = ",", header = 0, usecols = [4,5,6,11,12], names=('WTime', 'AccumulatedDis', 'VelocityAve', 'WalkVelocityAve', 'WalkVelocityDiv')) # csv読み込み
    walk_dist = my_gauss.MyGaussianDistribution(walk_df)

    # 検証データからセグメントの各データを取得
    segment_rest_df = pd.read_csv(savefile, sep = ",", header = 0, usecols = [3,5,6,9,10], names=('RTime', 'AccumulatedDis', 'VelocityAve', 'RestVelocityAve', 'RestVelocityDiv')) # csv読み込み
    segment_walk_df = pd.read_csv(savefile, sep = ",", header = 0, usecols = [4,5,6,11,12], names=('WTime', 'AccumulatedDis', 'VelocityAve', 'WalkVelocityAve', 'WalkVelocityDiv')) # csv読み込み
    
    # プロット
    rest_plot = my_plot.PlotUtility()
    walk_plot = my_plot.PlotUtility()
    rest_plot.scatter_plot(segment_rest_df['RTime'], segment_rest_df['RestVelocityAve'], [0], size=3)
    rest_plot.scatter_plot(rest_df['RTime'], rest_df['RestVelocityAve'], [1], size=6)
    rest_plot.scatter_plot([rest_dist.get_mean_vector()[0]], [rest_dist.get_mean_vector()[3]], [2], size=9)
    walk_plot.scatter_plot(segment_walk_df['WTime'], segment_walk_df['WalkVelocityAve'], [0], size=3)
    walk_plot.scatter_plot(walk_df['WTime'], walk_df['WalkVelocityAve'], [1], size=6)
    walk_plot.scatter_plot([walk_dist.get_mean_vector()[0]], [walk_dist.get_mean_vector()[3]], [2], size=9)
    rest_plot.show()
    walk_plot.show()

