import os, sys
import time
import datetime
import numpy as np
import pandas as pd
import pdb
import pickle

# 自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.interaction_analyzer as interaction_analyzer
from synchronization.graph_operation.graph_series import GraphSeriesAnalysis

# 自作ライブラリ
import cows.geography as geography
import synchronization.functions.utility as my_utility
from synchronization.detection_model.cut_point_search import cut_point_search, estimate_parameters

"""
行動同期を検出するプログラム（main関数）
"""

delta_c = 2 # コミュニティの抽出間隔 [minutes]
delta_s = 5 # データのスライス間隔 [seconds] 
epsilon = 10 # コミュニティ決定のパラメータ
dzeta = 12 # コミュニティ決定のパラメータ
leng = 1 # コミュニティ決定のパラメータ
cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
detection_record_file = "./synchronization/detection_model/test/"

def detect_interaction():
    """ 行動同期を検出する """
    start = datetime.datetime(2018, 5, 15, 0, 0, 0)
    end = datetime.datetime(2018, 5, 31, 0, 0, 0)
    target_list = ['20122','20129','20158','20170','20192','20197','20215','20267','20283'] # 2018/5/1 - 2018/7/31
    # target_list = ['20113', '20118', '20126', '20170', '20255', '20295', '20299'] # 2018/9/10 - 2018/12/25
    # target_list = ['20115', '20117', '20127', '20131', '20171', '20220', '20256', '20283', '20303'] # 2019/3/20 - 2019/7/3
    date = start
    while (date < end):
        s1 = time.time()
        interaction_graph_list = []
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        cow_id_list = com_creater.cow_id_list
        # --- インタラクショングラフを作成する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            print(t.strftime("%Y/%m/%d %H:%M:%S"))
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            interaction_graph_list.append(interaction_graph)
            t += datetime.timedelta(minutes=delta_c)
        e1 = time.time()
        print("処理時間", (e1-s1)/60, "[min]")
        # --- 対象牛のグラフ変化点を検知し，セッションを作る ---
        s2 = time.time()
        behavior_synch = com_creater.get_behavior_synch()
        position_synch = com_creater.get_position_synch()
        graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list, "Poisson")
        for cow_id in target_list:
            if (cow_id in cow_id_list):
                change_points, _ = graph_analyzer.detect_change_point(cow_id, 5, 5, threshold=400) # 変化点検知
                session_times = get_change_time(t_list, change_points)
                # --- 1つのセッションに対して，行動分岐点を探す ---
                for t_idx in session_times:
                    ses_start, ses_end = t_idx[0], t_idx[1]
                    interaction_graph = com_creater.make_interaction_graph(ses_start, ses_end, method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta)
                    community = com_creater.create_community(ses_start, ses_end, interaction_graph, delta=delta_s, leng=leng)
                    community = [com for com in community if str(cow_id) in com][0]
                    behaviors = behavior_synch.extract_df(ses_start, ses_end, delta_s) [community]
                    positions = position_synch.extract_df(ses_start, ses_end, delta_s) [community]
                    # --- セッション内の行動分岐点を探索する ---
                    change_point_series = cut_point_search(behaviors[str(cow_id)].values.tolist())
                    b_segments, p_segments = cut_data(behaviors, positions, change_point_series)
                    for b_seg, p_seg in zip(b_segments, p_segments):
                        theta = estimate_parameters(b_seg[str(cow_id)])
                        # 条件を満たしたセグメントは同期度をチェックする
                        if (not (theta[0] > 0.6 or theta[1] > 0.6) and theta[2] > 0.3):
                            score_dict = score_synchro(b_seg, p_seg, cow_id, community)
                            my_utility.write_values(detection_record_file + str(cow_id) + '_dict2.csv', [[b_seg.index[0], b_seg.index[-1], score_dict]])
                            my_utility.write_values(detection_record_file + str(cow_id) + '_max2.csv', [[b_seg.index[0], b_seg.index[-1], max(score_dict.values())]])
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        date += datetime.timedelta(days=1)
    return

def get_change_time(time_series, change_point_series):
    """ 変化点に従って, (start, end) のリストを作る """
    start_end_list = []
    change_idx = [i for i, flag in enumerate(change_point_series) if flag == 1]
    before_idx = 0
    for i in range(1, len(change_idx)):
        start = time_series[before_idx]
        end = time_series[change_idx[i]]
        start_end_list.append((start, end))
        before_idx = change_idx[i]
    return start_end_list

def cut_data(behaviors, positions, change_point_series):
    """ 変化点に従って，データを分割し，リスト化して返す """
    df_list1 = [] # behavior
    df_list2 = [] # position
    change_idx = [i for i, flag in enumerate(change_point_series) if flag == 1]
    before_idx = 0
    for idx in range(1, len(change_idx)):
        df_list1.append(behaviors[before_idx: change_idx[idx]])
        df_list2.append(positions[before_idx: change_idx[idx]])
        before_idx = change_idx[idx]
    df_list1.append(behaviors[change_idx[-1]:]) # 最後の変化点から最後まで
    df_list2.append(positions[change_idx[-1]:]) # 最後の変化点から最後まで
    return df_list1, df_list2

def score_synchro(beh_df, pos_df, target_cow_id, community, dis_threshold=10):
    """ 同期をスコア化する """
    score_dict = {} # 返却値
    score_matrix = np.eye(3)
    target_beh = beh_df[str(target_cow_id)].values
    target_pos = pos_df[str(target_cow_id)].values
    if (target_cow_id in community):
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
    detect_interaction()