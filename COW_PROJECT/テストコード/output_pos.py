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
import behavior_classification.functions.regex as regex

if __name__ == '__main__':
    # --- 変数定義 ---
    filename = "csv/20158.csv"
    start = datetime.datetime(2018, 12, 30, 0, 0, 0)
    end = datetime.datetime(2018, 12, 31, 0, 0, 0)
    target_cow_id = 20158

    # テスト用の1日のデータを読み込み
    time_list, position_list, distance_list, velocity_list, angle_list = loading.load_gps(target_cow_id, start, end) #2次元リスト (1日分 * 日数分)
    if (len(position_list[0]) != 0):
        for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
            # ---前処理---
            t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
            if (os.path.exists(filename)): # ファイルがすでに存在しているとき
                #####出力#####
                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(("Time", "latitude", "longitude", "velocity"))
                    for time, (lat,lon), vel in zip(t_list, p_list, v_list):
                        writer.writerow([time, lat, lon, vel])
            else:  # ファイルが存在していないとき
                #####出力#####
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(("Time", "latitude", "longitude", "velocity"))
                    for time, (lat,lon), vel in zip(t_list, p_list, v_list):
                        writer.writerow([time, lat, lon, vel])