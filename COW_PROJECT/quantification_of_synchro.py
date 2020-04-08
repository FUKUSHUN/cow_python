#-*- encoding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import math
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
import cows.cowshed as Cowshed
import cows.cows_community as comm
import behavior_classification.functions.loading as loading
import behavior_classification.functions.preprocessing as preprocessing
import behavior_classification.functions.plotting as plotting
import behavior_classification.functions.analyzing as analyzing
import behavior_classification.functions.regex as regex
import behavior_classification.functions.postprocessing as postprocessing
import behavior_classification.functions.output_features as output_features
import behavior_synchronization.functions.synchronization_method as synchmethod
import behavior_synchronization.functions.dtw as dtw

# 別ファイルでモジュール化
def get_existing_cow_list(date:datetime, filepath):
    """ 引数の日にちに第一放牧場にいた牛のリストを得る """
    filepath = filepath + date.strftime("%Y-%m") + ".csv"
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if (datetime.datetime.strptime(row[0], "%Y/%m/%d") == date):
                return row[1:]

    print("指定の日付の牛のリストが見つかりません", date.strftime("%Y/%m/%d"))
    sys.exit()

# 別ファイルでモジュール化
def get_behavior_df(date, cow_id, model1, model2):
    """ 個々の牛の1日分の行動分類を行う """
    print(sys._getframe().f_code.co_name, "実行中")
    print("行動分類を行います---：日付: ", date.strftime("%Y/%m/%d"), ", 個体番号: ", cow_id)
    filename = "behavior_classification/training_data/features.csv"
    existed = output_features.output_features(filename, date, cow_id)
    if (existed): # GPSデータが存在していれば
        ### 各特徴から行動を分類する ###
        df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,1,2,3,4,5,6,7,9], names=('Time', 'RCategory', 'WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance')) # csv読み込み
        labels = []
        probs = []
        x1, x2, x3, x4, x5, x6, x7, x8 = df['RCategory'].tolist(), df['WCategory'].tolist(), df['RTime'].tolist(), df['WTime'].tolist(), df['AccumulatedDis'].tolist(), df['Velocity'].tolist(), df['MVelocity'].tolist(), df['Distance'].tolist()
        x = np.array((x1, x2, x3, x4, x5, x6, x7, x8)).T
        result1, prob1 = model1.predict(x), model1.predict_proba(x) # 休息セグメントの結果と確率
        result2, prob2 = model2.predict(x), model2.predict_proba(x) # 活動セグメントの結果と確率
        for a, b, c, d in zip(result1, result2, prob1, prob2):
            labels.append(a)
            labels.append(b)
            probs.append(np.insert(c, 2, 0.0))
            probs.append(d)
        # --- 復元 ---
        t_list = loading.make_time_list(date)
        zipped_t_list = regex.str_to_datetime(df['Time'].tolist())
        times, labels = postprocessing.decompress(t_list, zipped_t_list, labels)
        df = cut_used_time(times, labels, cow_id)
        return df
    else:
        t_list = loading.make_time_list(date)
        labels = [-1 for l in range(len(t_list))]
        df = cut_used_time(t_list, labels, cow_id)        
        return df


# 別ファイルでモジュール化
def cut_used_time(time_list, label_list, cow_id):
    """ コミュニティ同期に使用する部分を抽出する """
    new_t_list = []
    new_l_list = []
    for t, l in zip(time_list, label_list):
        time_num = int(t.strftime("%H%M"))
        if (time_num < 830 or 1230 <= time_num): # 12:30-8:30に限定
            new_t_list.append(t)
            new_l_list.append(l)
    df = pd.DataFrame({"Time":new_t_list, str(cow_id):new_l_list}).set_index("Time")
    return df


if __name__ == '__main__':
    # --- 分析用のファイル ---
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/"
    model_filename1 = "behavior_classification/bst/model.pickle"
    model_filename2 = "behavior_classification/bst/model2.pickle"
    # --- 分析を行う期間（この期間中1日ずつ分析を行う） ---
    start = datetime.datetime(2018, 10, 15, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    end = datetime.datetime(2018, 10, 25, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    # --- コミュニティ生成間隔 [minutes] ---
    community_interval = 5
    print(os.getcwd())

    ### 1日ずつ検証 ###
    record_list = [] # 結果格納用リスト
    the_day = start
    while (the_day < end):
        data_list = pd.DataFrame([])
        cow_id_list = get_existing_cow_list(the_day, cows_record_file) # その日に放牧場にいた牛の個体番号のリストを取得
        model1 = joblib.load(model_filename1) # モデルのロード
        model2 = joblib.load(model_filename2) # モデルのロード
        ### 1頭ずつその日の行動を得る ###
        for cow_id in cow_id_list:
            behavior_data = get_behavior_df(the_day, cow_id, model1, model2)
            data_list = pd.concat([data_list, behavior_data], axis=1)
        #data_list.to_csv("behavior_synchronization/prediction.csv")

        ### 5分ごとにコミュニティを分析する ###
        cows = Cowshed.Cowshed(the_day) # その日の牛の集合
        dt = the_day + datetime.timedelta(hours=9) # その日の始まり
        while (dt < the_day + datetime.timedelta(hours=9) + datetime.timedelta(days=1)):
            time_num = int((dt.strftime("%H%M")))
            time_list = []
            score_list = []
            if (time_num < 830 or 1230 <= time_num):
                position_df = cows.get_cow_list(dt, dt + datetime.timedelta(minutes = community_interval) - datetime.timedelta(seconds = 1)) # 1日のデータから5分分の位置情報を取得
                behavior_df = data_list.loc[dt : dt + datetime.timedelta(minutes = community_interval) - datetime.timedelta(seconds = 1)] # 1日のデータから5分分の行動分類を取得
                nodes_list = comm.extract_community(position_df, 10) #コミュニティ生成
                
                for i in range(len(position_df.columns)):
                    cow_id = position_df.iloc[0,i]
                    com = [] # その牛のコミュニティメンバー（自分を含む）
                    com_data = [] # その牛のコミュニティメンバーの行動データ（自分を含む）
                    if (cow_id == 20299):
                        for team in nodes_list:
                            if(cow_id in team):
                                com = team
                                break
                        for column_name, behavior in behavior_df.iteritems():
                            cow_id = column_name
                            if (int(cow_id) in com):
                                com_data.append(behavior)
                            if (int(cow_id) == 20158):
                                main_cow_behavior = behavior
                                behavior_vector = synchmethod.make_vector(behavior)
                        #score = synchmethod.average_behavior(com, com_data)
                        #score_list.append(score)
                        scores = []
                        for cow_id, behavior in zip(com, com_data):
                            if (cow_id != 20158):
                                f = dtw.dtw(main_cow_behavior, behavior)
                                score = dtw.get_dtw_sim(f, main_cow_behavior, behavior)
                                scores.append(score)
                        average_score = sum(scores) / len(scores) if (len(scores) != 0) else np.nan
                        all_elem = [dt.strftime("%m/%d %H:%M")]
                        all_elem.append(len(com))
                        all_elem.extend(behavior_vector)
                        all_elem.append(average_score)
                        record_list.append(all_elem)
                        print(all_elem)
                        
                """
                plt.figure()
                for column_name, behavior in behavior_df.iteritems():
                    plt.plot(range(0,len(behavior)), behavior)
                plt.show()
                sys.exit()
                """
            else:
                all_elem = [dt.strftime("%m/%d %H:%M")]
                record_list.append(all_elem)
            time_list.append(dt)
            dt += datetime.timedelta(minutes=community_interval) # 5分進める

        the_day += datetime.timedelta(days=1) # 1日進める
    with open("behavior_synchronization/20158.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Community size", "Prob_rest", "Prob_graze", "Prob_walk", "DTW_score"])
        for row in record_list:
            writer.writerow(row)
            
        
