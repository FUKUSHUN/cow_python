import os, sys
import csv
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

# 自作クラス
os.chdir('../../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater

# 自作メソッド
import cows.geography as geography
import synchronization.functions.utility as my_utility
from synchronization.detection_model.cut_point_search import cut_point_search, estimate_parameters

"""
recordにある牛の行動を可視化する
"""
def load_csv(filename):
    """ CSVファイルをロードし，記録内容をリストで返す """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        records = [row for row in reader]
    return records

def fetch_information(date):
    """ 1日分の牛のデータを読み込む """
    date = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
    cow_id_list = com_creater.cow_id_list
    return cow_id_list, com_creater

def cut_data(behaviors, positions, change_point_series):
    """ 変化点に従って，データを分割し，リスト化して返す """
    df_list1 = [] # behavior
    df_list2 = [] # position
    change_idx = [i for i, flag in enumerate(change_point_series) if flag == 1]
    before_idx = change_idx[0]
    for idx in range(1, len(change_idx)):
        df_list1.append(behaviors[before_idx: change_idx[idx]])
        df_list2.append(positions[before_idx: change_idx[idx]])
        before_idx = change_idx[idx]
    df_list1.append(behaviors[change_idx[-1]:]) # 最後の行動分岐点から最後まで
    df_list2.append(positions[change_idx[-1]:]) # 最後の行動分岐点から最後まで
    return df_list1, df_list2

def score_synchro(beh_df, pos_df, target_cow_id, community, dis_threshold=10):
    """ 同期をスコア化する """
    score_dict = {} # 返却値
    score_matrix = np.eye(3)
    target_beh = beh_df[str(target_cow_id)].values
    target_pos = pos_df[str(target_cow_id)].values
    community.remove(target_cow_id)
    for cow_id in community:
        score = 0
        nearcow_pos = pos_df[cow_id].values
        nearcow_beh = beh_df[cow_id].values
        for i in range(len(target_beh)):
            lat1, lon1 = target_pos[i][0], target_pos[i][1]
            lat2, lon2 = nearcow_pos[i][0], nearcow_pos[i][1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            # 近い距離にいれば同期しているかを確認する
            if (dis <= dis_threshold):                
                score += score_matrix[target_beh[i], nearcow_beh[i]]
        score_dict[cow_id] = score
    return score_dict

if __name__ == "__main__":
    delta_s = 5 # データのスライス間隔 [seconds] 
    epsilon = 10 # コミュニティ決定のパラメータ
    dzeta = 12 # コミュニティ決定のパラメータ
    leng = 1 # コミュニティ決定のパラメータ
    record_file = "./synchronization/detection_model/record.csv"
    output_file = "./synchronization/detection_model/"
    records = load_csv(record_file)
    for i, row in enumerate(records):
        if (int(row[5]) == 1):
            # 牛の行動をファイルからロード
            target_cow_id = row[4][:5]
            date = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S") - datetime.timedelta(hours=9) # 翌朝の可能性もあるので時間を9時間戻す
            cow_id_list, com_creater = fetch_information(date)
            
            # 変化点部分の行動を切り取る
            start = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S")
            end = datetime.datetime.strptime(row[2] + " " + row[3], "%Y/%m/%d %H:%M:%S") + datetime.timedelta(seconds=5) # 不等号で抽出するため5秒追加
            interaction_graph = com_creater.make_interaction_graph(start, end, method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta)
            community = com_creater.create_community(start, end, interaction_graph, delta=delta_s, leng=leng)
            community = [com for com in community if str(target_cow_id) in com][0]

            # 行動分岐点を探す
            behaviors = com_creater.get_behavior_synch().extract_df(start, end, delta_s)
            positions = com_creater.get_position_synch().extract_df(start, end, delta_s)
            behaviors = behaviors[community]
            positions = positions[community]
            change_point_series = cut_point_search(behaviors[str(target_cow_id)].values.tolist())
            b_segments, p_segments = cut_data(behaviors, positions, change_point_series)
            for b_seg, p_seg in zip(b_segments, p_segments):
                theta = estimate_parameters(b_seg[str(target_cow_id)])
                # 条件を満たしたセグメントは同期度をチェックする
                if (not (theta[0] > 0.6 or theta[1] > 0.6) and theta[2] > 0.3):
                    score_synchro(b_seg, p_seg, target_cow_id, community)
                    pdb.set_trace()
            
