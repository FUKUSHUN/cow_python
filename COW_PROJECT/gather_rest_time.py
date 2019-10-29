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
import cows.geography as geo
import behavior_classification.hmm as hmm
import behavior_classification.loading as loading
import behavior_classification.preprocessing as preprocessing
import behavior_classification.plotting as plotting
import behavior_classification.analyzing as analyzing
import behavior_classification.regex as regex
import behavior_classification.postprocessing as postprocessing
import behavior_classification.output_features as output_features
import image.adjectory_image as disp

"""
基本的な手順はoutput_features.output_features()に集約されているが，個々のグラフなどの可視化にはこちらのメインを活用
"""

if __name__ == '__main__':
    filename = "behavior_classification/training_data/features.csv"
    start = datetime.datetime(2018, 5, 1, 0, 0, 0)
    end = datetime.datetime(2018, 5, 31, 0, 0, 0)
    sum_rest_list = []
    print(os.getcwd())
    time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(20158, start, end) #2次元リスト (1日分 * 日数分)
    if (len(position_list[0]) != 0):
        for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
            # ---前処理---
            t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
            
            # 畳み込み
            v_list = preprocessing.convolution(v_list, 3)
            d_list = preprocessing.convolution(d_list, 3)
            t_list = preprocessing.elimination(t_list, 3)
            p_list = preprocessing.elimination(p_list, 3)
            a_list = preprocessing.elimination(a_list, 3)

            # 圧縮操作
            zipped_list = output_features.compress(t_list, p_list, d_list, v_list) # 圧縮する

            # ---特徴抽出---
            output_features.output_feature_info(filename, [row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
            
            # 各特徴
            df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,1,2,3,4,5,6,7,9], names=('Time', 'RCategory', 'WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance')) # csv読み込み
            print(df['AccumulatedDis'])
            
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
                if (c.max() >= 0.7 and a == 0):
                    labels.append(a)
                else:
                    labels.append(1)

            # --- 復元 ---
            zipped_t_list = regex.str_to_datetime(df['Time'].tolist())
            new_t_list, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
            sum_rest = 0
            for i in labels:
                if (i == 0):
                    sum_rest += 5
            sum_rest_list.append(sum_rest)
            