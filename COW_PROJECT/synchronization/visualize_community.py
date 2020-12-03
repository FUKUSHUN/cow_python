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
import synchronization.functions.utility as my_utility
from synchronization.graph_operation.graph_series import GraphSeriesAnalysis
from synchronization.set_operation.set_series import SetSeriesAnalysis
from synchronization.topic_model.lda import GaussianLDA

# 自作ライブラリ
import synchronization.topic_model.make_session as make_session
import synchronization.topic_model.session_io as session_io

delta_c = 2 # コミュニティの抽出間隔 [minutes]
delta_s = 5 # データのスライス間隔 [seconds] 
epsilon = 12 # コミュニティ決定のパラメータ
dzeta = 12 # コミュニティ決定のパラメータ
leng = 1 # コミュニティ決定のパラメータ
cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル

def visualize_community(date):
    s1 = time.time()
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
    cow_id_list = com_creater.cow_id_list
    # --- 行動同期を計測する ---
    t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
    t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
    t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
    while (t < t_end):
        interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
            if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
        community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, delta=delta_s, leng=leng) \
            if (t_start <= t) else [[]] # コミュニティを決定
        com_creater.visualize_position(t, t+datetime.timedelta(minutes=delta_c), community, target_cow_id=target_cow_id, delta=delta_s) # 位置情報とコミュニティをプロット
        t += datetime.timedelta(minutes=delta_c)
    e1 = time.time()
    print("処理時間", (e1-s1)/60, "[min]")
    return

if __name__ == "__main__":
    target_cow_id = '20303'
    date = datetime.datetime(2019, 3, 21, 0, 0, 0)
    visualize_community(date) # 可視化ムービーの作成
