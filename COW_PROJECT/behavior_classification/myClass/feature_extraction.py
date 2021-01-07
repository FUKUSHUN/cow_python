#-*- encoding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import csv
import datetime
import sys
import os
import statistics
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from sklearn.externals import joblib
import pickle

# 自作クラス
added_path = os.path.abspath('../')
sys.path.append(added_path)
import cows.geography as geo
import behavior_classification.functions.hmm as hmm
import behavior_classification.functions.loading as loading
import behavior_classification.functions.preprocessing as preprocessing
import behavior_classification.functions.plotting as plotting
import behavior_classification.functions.analyzing as analyzing
import behavior_classification.functions.regex as regex
import behavior_classification.functions.postprocessing as postprocessing

""" このコードではfunctions/output_features.pyの関数群をクラスとして管理することを目標に作成されました """
class FeatureExtraction:
    date = None # 特徴が作成されるデータの日にち
    cow_id = None # 特徴が作成されるデータの牛のID

    def __init__(self, date:datetime, cow_id):
        self.date = date
        self.cow_id = cow_id

    def compress(self, t_list, p_list, d_list, v_list):
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
                if (self._choice_state(velocity) == 0):
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
                if (self._choice_state(velocity) == 1):
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
                if (self._choice_state(velocity) == 2):
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

            if (self._choice_state(velocity) == 0):
                is_rest = True
                is_graze = False
                is_walk = False
            elif (self._choice_state(velocity) == 1):
                is_rest = False
                is_graze = True
                is_walk = False
            else:
                is_rest = False
                is_graze = False
                is_walk = True
        
        # 最後の行動を登録して登録終了
        zipped_list.append(((start, end), p_tmp_list, d_tmp_list, v_tmp_list, self._choice_state(velocity)))
        print("---圧縮が終了しました")
        print(sys._getframe().f_code.co_name, "正常終了\n")
        return zipped_list


    def _choice_state(self, velocity, r_threshold = 0.0694, g_threshold = 0.181):
        """ 休息・採食・歩行を判断する（今は速度データをもとに閾値や最近傍のアプローチだが変更する可能性あり）"""
        if (velocity < r_threshold):
            return 0 # 休息
        elif (r_threshold <= velocity and velocity < g_threshold):
            return 1 # 採食
        else:
            return 1 # 歩行 (実施不透明)


    def output_feature_info(self, t_list, p_list, d_list, v_list, l_list):
        """ 特徴をCSVにして出力する (圧縮が既に行われている前提) 
        Parameter
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
        rest_vel = None

        #####登録情報#####
        time_index = None
        resting_time_category = None # 時間帯のカテゴリ (日の出・日の入時刻を元に算出)
        walking_time_category = None # 時間帯のカテゴリ (日の出・日の入時刻を元に算出)
        previous_rest_length = None #圧縮にまとめられた前の休息の観測の個数
        walking_length = None #圧縮にまとめられた歩行の観測の個数
        moving_distance = None #休息間の距離
        mean_vel = None # 両セグメント内の平均速度
        resting_velocity_average = None # 停止セグメントの速度の平均
        resting_velocity_deviation = None # 停止セグメントの速度の標準偏差
        walking_velocity_average = None # 活動セグメントの速度の平均
        walking_velocity_deviation = None # 停止セグメントの速度の標準偏差
        moving_direction = None #次の休息への移動方向

        print("特徴を計算します---")
        feature_list =[]
        behavior_dict = {"resting":0, "walking":1} # 行動ラベルの辞書
        initial_datetime = t_list[0][0]
        #####登録#####
        for i, (time, pos, dis, vel, label) in enumerate(zip(t_list, p_list, d_list, v_list, l_list)):
            if (label == behavior_dict["walking"]): # 歩行
                if (i != 0): # 最初は休息から始まるようにする (もし最初が歩行ならそのデータは削られる)
                    time_index.append((time[0], time[1]))
                    walking_time_category = self._decide_time_category(time[0], initial_datetime)
                    walking_length = (time[1] - time[0]).total_seconds() / 5 + 1
                    walking_velocity_average, walking_velocity_deviation = self._extract_mean(vel) # 活動セグメント内の平均速度とその標準偏差
                    vel.extend(rest_vel) # 休息時の速度のリストと結合
                    max_vel = max(vel)
                    min_vel = min(vel)
                    mean_accumulated_distance, _ = self._extract_mean(dis) # 行動内での移動距離の１観測あたり
                    mean_vel, _ = self._extract_mean(vel)

            if (label == behavior_dict["resting"]): # 休息
                ###前後関係に着目した特徴の算出###
                after_lat = pos[0][0]
                after_lon = pos[0][1]
                resting_time_category = self._decide_time_category(time[0], initial_datetime)
                if (before_lat is not None and before_lon is not None and rest_vel is not None):
                    moving_distance, moving_direction = geo.get_distance_and_direction(before_lat, before_lon, after_lat, after_lon, True) #前の重心との直線距離
                    
                    ###リストに追加###
                    feature_list.append([time_index, resting_time_category, walking_time_category, previous_rest_length, walking_length, mean_accumulated_distance, mean_vel, max_vel, min_vel, resting_velocity_average, resting_velocity_deviation, walking_velocity_average, walking_velocity_deviation, moving_distance, moving_direction])

                ###引継###
                previous_rest_length = (time[1] - time[0]).total_seconds() / 5 + 1
                before_lat = pos[len(pos) - 1][0]
                before_lon = pos[len(pos) - 1][1]
                resting_velocity_average, resting_velocity_deviation = self._extract_mean(vel) # 休息セグメント内の平均速度とその標準偏差
                rest_vel = vel
                time_index = [(time[0], time[1])]
        
        print("---特徴を計算しました")
        print(sys._getframe().f_code.co_name, "正常終了\n")
        return feature_list

    def _decide_time_category(self, dt, date):
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


    def _extract_mean(self, some_list):
        """ リスト内の平均と標準偏差を求める
        Parameter
            some_list	: なにかのリスト """
        ave = 0.0 # 平均
        dev = 0.0 # 標準偏差
        try:
            ave = statistics.mean(some_list)
            dev = statistics.stdev(some_list)
        except statistics.StatisticsError:
            pass
        return ave, dev


    def output_features(self):
        """ 日付と牛の個体番号からその日のその牛の位置情報を用いて特徴のファイル出力を行う
        Parameters
            filename	: 保存するファイルの絶対パス
            date	: 日付	: datetime
            cow_id	: 牛の個体番号．この牛の特徴を出力する """	
        t_list, p_list, d_list, v_list, a_list = loading.load_gps(self.cow_id, self.date) # 1日分のデータ
        # ---前処理---
        t_list, p_list, d_list, v_list, a_list = loading.select_used_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
        if (len(p_list) != 0 and len(d_list) != 0 and len(v_list) != 0): # データがない場合は飛ばす
            
            # 畳み込み
            #v_list = preprocessing.convolution(v_list, 3)
            #d_list = preprocessing.convolution(d_list, 3)
            #t_list = preprocessing.elimination(t_list, 3)
            #p_list = preprocessing.elimination(p_list, 3)
            #a_list = preprocessing.elimination(a_list, 3)

            # 圧縮操作
            zipped_list = self.compress(t_list, p_list, d_list, v_list) # 圧縮する

            # ---特徴抽出---
            feature_list = self.output_feature_info([row[0] for row in zipped_list], [row[1] for row in zipped_list], [row[2] for row in zipped_list], [row[3] for row in zipped_list], [row[4] for row in zipped_list]) # 特徴を出力する
            return feature_list # 特徴出力に成功したのでTrueを返す
        else:
            return [] # データがなければFalseを返す