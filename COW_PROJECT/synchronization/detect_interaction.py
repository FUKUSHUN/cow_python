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
    behavior_model = pickle.load(open('./synchronization/detection_model/model.pkl', 'rb'))
    start = datetime.datetime(2018, 6, 17, 0, 0, 0)
    end = datetime.datetime(2018, 6, 22, 0, 0, 0)
    # target_list = ['20122', '20158', '20192', '20215']
    target_list = ['20122', '20126', '20129', '20158' ,'20192', '20197', '20215', '20220', '20267', '20283'] # 2018/5/1 - 2018/7/31
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
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="behavior", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
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
                for i, t_idx in enumerate(session_times):
                    ses_start, ses_end = t_idx[0], t_idx[1]
                    interaction_graph = com_creater.make_interaction_graph(ses_start, ses_end, method="behavior", delta=delta_s, epsilon=epsilon, dzeta=dzeta)
                    community = com_creater.create_community(ses_start, ses_end, interaction_graph, delta=delta_s, leng=leng)
                    community = [com for com in community if str(cow_id) in com][0]
                    # density = calculate_graph_density(community, interaction_graph, cow_id_list)
                    behaviors = behavior_synch.extract_df(ses_start, ses_end, delta_s) [community]
                    positions = position_synch.extract_df(ses_start, ses_end, delta_s) [community]
                    # --- セッション内の行動分岐点を探索する ---
                    change_point_series = cut_point_search(behaviors[str(cow_id)].values.tolist())
                    b_segments, p_segments = cut_data(behaviors, positions, change_point_series)
                    score_dict = {}
                    for c in cow_id_list:
                        score_dict[c] = 0 # 牛のIDをキーにスコアを格納する
                    for b_seg, p_seg in zip(b_segments, p_segments):
                        theta = estimate_parameters(b_seg[str(cow_id)])
                        # 条件を満たしたセグメントは同期度をチェックする
                        pattern = np.argmax(behavior_model.predict([theta * len(b_seg[str(cow_id)])]))
                        if (not (pattern == 0 or pattern == 1)):
                            behavior_seg_df = behavior_synch.extract_df(b_seg[str(cow_id)].index[0], b_seg[str(cow_id)].index[-1], delta_s)
                            score_matrix = get_score_matrix(behavior_seg_df)
                            scores = score_synchro(b_seg, p_seg, cow_id, community, score_matrix)
                            for key in scores.keys():
                                score_dict[key] += scores[key]
                    try:
                        my_utility.write_values(detection_record_file + str(cow_id) + start.strftime('_%Y%m%d') + '.csv', [[ses_start, ses_end, score_dict, max(score_dict.values()), (ses_end - ses_start).total_seconds(), i]])
                    except:
                        print('error')
                        continue
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

# def calculate_graph_density(community, graph, cow_id_list):
#     """ ある牛の所属するコミュニティのグラフ密度を算出する (グラフ全体の重みの平均以上の場合エッジを引く) """
#     # グラフ全体の重みの平均以上の場合エッジを引いたグラフにする
#     N = len(graph) # 頭数
#     average_weight = np.sum (graph) / (N * N)
#     unweighted_graph = np.zeros((N, N))
#     for i in range(len(unweighted_graph)):
#         for j in range(len(unweighted_graph)):
#             unweighted_graph[i, j] = 1 if graph[i, j] >= average_weight else 0
#     density = 0
#     # targetの所属するコミュニティのメンバのみからなるグラフに成形する
#     index_list = [cow_id_list.index(str(cow_id)) for cow_id in community] # インタラクショングラフの何行目かのインデックス
#     index_list = sorted(index_list)
#     formed_graph = _form_graph(unweighted_graph, index_list)
#     # グラフの密度を算出する
#     density = _calculate_graph_density(formed_graph)
#     return density

# def _form_graph(W, index_list):
#     """ 全体のグラフ（行列）からインデックス行（列）のみを取り出して成形する """
#     graph_rows = [W[index] for index in index_list]
#     formed_graph = np.stack(graph_rows, axis=0)
#     graph_columns = [formed_graph[:,index] for index in index_list]
#     formed_graph = np.stack(graph_columns, axis=1)
#     return formed_graph

# def _calculate_graph_density(W):
#     """ グラフの密度を求める
#         W:  コミュニティメンバのみに成形後のインタラクショングラフ """
#     # 辺の数（重みが0より大きい）を数え上げる
#     K = len(W)
#     count = 0
#     for i in range(K):
#         for j in range(K):
#             if (i < j and 0 < W[i,j]):
#                 count += 1
#     # 完全グラフの時の辺の数で割り密度を算出する
#     density = count / (K * (K-1) / 2) if 1 < K else 0
#     return density

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

def get_score_matrix(beh_df):
    """ スコア行列を獲得する """
    score_matrix = np.zeros((3, 3))
    count_vector = np.zeros(3)
    for i in range(3):
        count_vector[i] = (beh_df == i).values.sum()
    for i in range(3):
        score_matrix[i, i] = (count_vector[(i+1)%3] + count_vector[(i+2)%3]) / (count_vector.sum() * 2)
    return score_matrix

def score_synchro(beh_df, pos_df, target_cow_id, community, score_matrix, dis_threshold=10):
    """ 同期をスコア化する """
    score_dict = {} # 返却値
    # score_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
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
            if (dis <= dis_threshold and _check_position(lat1, lon1) and _check_position(lat2, lon2)):
                score += score_matrix[target_beh[i], nearcow_beh[i]]
        score_dict[cow_id] = score / 12 # 1分間あたりになおす
    return score_dict

def _check_position (lat, lon):
    """ 牛舎や屋根下にいないかチェック """
    cowshed_boundary = (34.882449, 134.863557) # これより南は牛舎
    roof_boundary = [(34.882911, 134.863357), (34.882978, 134.863441)] # 肥育横の屋根の座標（南西，北東）
    if (lat < cowshed_boundary[0]):
        return False
    if (roof_boundary[0][0] < lat and lat < roof_boundary[1][0] and roof_boundary[0][1] < lon and lon < roof_boundary[1][1]):
        return False
    return True

if __name__ == "__main__":
    detect_interaction()