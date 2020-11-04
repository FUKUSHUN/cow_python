import os, sys
import time
import datetime
import numpy as np
import pandas as pd
import pdb
import pickle

""" トピックモデルを適用するために入力データ（ドキュメント集合, corpus）を作成する関数群 """

def get_focused_community(communities_list, target_cow_id):
    """ 各時刻でのコミュニティの集合からターゲットの牛の所属するコミュニティのみを取り出す """
    community_list = []
    for communities in communities_list:
        for community in communities:
            if (str(target_cow_id) in community):
                community_list.append(community)
                break
    return community_list

def process_time_series(time_series, communities, change_points):
    """ 変化点に1, それ以外に0を立てたリストから (cow_idを単語とする) セッションを作る """
    session_list = []
    is_first = True
    for time, community, is_changed in zip(time_series, communities, change_points):
        if (is_first):
            session = {time:community} # definition
            is_first = False
        elif (is_changed == 1):
            session_list.append(session)
            session = {time:community} # definition            
        else:
            session[time] = community # addition
    return session_list

def exchange_cowid_to_space(id_session, behavior_synch, delta_c, delta_s, dim=3):
    """ cow_idを単語としたドキュメント集合を特徴表現に直す """
    space_session = []
    for ses in id_session:
        space_expression = []
        for key in ses:
            extracted_df = behavior_synch.extract_df(key, key+datetime.timedelta(minutes=delta_c), delta_s)
            for cow_id in ses[key]:
                prop_vec = _measure_behavior_ratio(extracted_df[cow_id].values)
                space_expression.append(prop_vec[:dim]) # 2次元で登録したいときと3次元で登録したいとき用に[:dim]とする
        space_session.append(np.array(space_expression))
    return space_session

def _measure_behavior_ratio(arraylist):
        """ 行動割合を算出する """
        behavior_0, behavior_1, behavior_2 = 0, 0, 0 # カウントアップ変数
        # 各時刻の行動を順番に走査してカウント
        for elem in arraylist:
            if (elem == 0):
                behavior_0 += 1
            elif (elem == 1):
                behavior_1 += 1
            else:
                behavior_2 += 1
        length = len(arraylist)
        proportion_b1 = behavior_0 / length
        proportion_b2 = behavior_1 / length
        proportion_b3 = behavior_2 / length
        prop_vec = np.array([proportion_b1, proportion_b2, proportion_b3])
        return prop_vec

def restore_time_series(time_series, change_points, additional_info):
    """ 変化点ごとに圧縮されたsession_listごとに得られた情報を元のtime_seriesに還元する """
    additional_info_time_series = []
    for info in additional_info:
        info_time_series = []
        i = 0 # info (arraylike) のインデックス
        for _, is_changed in zip(time_series, change_points):
            if (is_changed):
                i += 1
                info_time_series.append(info[i])
            else:
                info_time_series.append(info[i])
        additional_info_time_series.append(info_time_series)
    return additional_info_time_series