#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cows_relation as cr #自作クラス
import gps.gps_nmea_data_list as gpslist #自作クラス

"""
    Louvainアルゴリズムを用いてコミュニティを生成する
    df :    pandas.DataFrame    :牛の個体番号とGPSデータリストを所持
    dfの形式はcowshed.Cowshed.get_cow_list()を参照すること
    javaのプログラムをそのまま移植している部分が大きいです
"""
def extract_community(df:pd.DataFrame, threshold):
    matrix = np.zeros(len(df.columns), len(df.columns)) # 行列の作成
    cnt = 0
    ave = 0.0
    for i in range(len(df.cow_data.columns)):
        for j in range(len(df.cow_data.columns)):
            if(i != j):
                tcr = cr.TwoCowsRelation(df.cow_data.iloc[1,i], df.cow_data.iloc[1,j])
                value = tcr.count_near_distance_time(threshold)
                matrix[i,j] = value
                ave += value # sum
                cnt += 1
    ave = ave / cnt #average
    dev = 0.0 #standard deviation
    for i in range(len(df.cow_data.columns)):
        for j in range(len(df.cow_data.columns)):
            if(i != j):
                dev += (matrix[i,j] - ave) ** 2
    dev = dev ** (1 / 2)
    for i in range(len(df.cow_data.columns)):
        for j in range(len(df.cow_data.columns)):
            if(10 < (matrix[i, j] - ave) / dev): #偏差値60以上と同義
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0