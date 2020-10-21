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
leng = 5 # コミュニティ決定のパラメータ
start = datetime.datetime(2018, 10, 10, 0, 0, 0)
end = datetime.datetime(2018, 10, 25, 0, 0, 0)
target_list = ['20113','20170','20295','20299']
cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
change_point_file = "./synchronization/change_point/"
corpus_file = "./synchronization/topic_model/corpus/"
communities_list = []
interaction_graph_list = []
corpus = []

def create_corpus():
    date = start
    while (date < end):
        s1 = time.time()
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="behavior", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, delta=delta_s, leng=leng) \
                if (t_start <= t) else [[]] # コミュニティを決定
            # com_creater.visualize_position(t, t+datetime.timedelta(minutes=delta_c), community, target_cow_id='20113', delta=delta_s) # 位置情報とコミュニティをプロット1
            interaction_graph_list.append(interaction_graph)
            communities_list.append(community)
            t += datetime.timedelta(minutes=delta_c)
        e1 = time.time()
        print("処理時間", (e1-s1)/60, "[min]")
        # --- 変化点を検知し，セッションを作る ---
        s2 = time.time()
        behavior_synch = com_creater.get_behavior_synch()
        set_analyzer = SetSeriesAnalysis(cow_id_list, communities_list)
        for cow_id in target_list:
            change_points = set_analyzer.detect_change_point(cow_id, t_list)
            df = pd.concat([pd.Series(t_list), pd.Series(change_points)], axis=1, names=["time", "change_flag"])
            df.to_csv("./synchronization/set_operation/"+ str(cow_id) + ".csv")
            community_list = make_session.get_focused_community(communities_list, cow_id)
            cow_id_session = make_session.process_time_series(t_list, community_list, change_points)
            space_session = make_session.exchange_cowid_to_space(cow_id_session, behavior_synch, delta_c, delta_s)
            corpus.extend(space_session)
            session_io.write_session(space_session, corpus_file+cow_id + "/" + date.strftime("%Y%m%d/"))
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        date += datetime.timedelta(days=1)
    return

def load_corpus():
    s1 = time.time()
    global corpus
    # --- ファイルを読み込みコーパスを作成する ---
    date = start
    while (date < end):
        for cow_id in target_list:
            space_session = session_io.read_session(corpus_file+cow_id + "/" + date.strftime("%Y%m%d/"))
            corpus.extend(space_session)
        date += datetime.timedelta(days=1)
    e1 = time.time()
    print("処理時間", (e1-s1)/60, "[min]")
    return


if __name__ == "__main__":
    is_create = True
    is_load = True
    if (is_create):
        create_corpus()
    if (is_load):
        load_corpus()

    # LDAのハイパーパラメータ設定
    alpha = np.array([1, 1, 1, 1]) # parameter for dirichlet
    psi = np.array([[1, 0], [0, 1]]) # parameter for Gaussian Wishert
    m = np.array([0.4, 0.4]) # parameter for Gaussian Wishert
    nu = 1 # parameter for Gaussian Wishert
    beta = 1 # parameter for Gaussian Wishert
    max_iter = 3000

    # ギブスサンプリングによるクラスタリング
    gaussian_lda = GaussianLDA(corpus = corpus, num_topic=4, dimensionality=2)
    Z = gaussian_lda.inference(alpha, psi, nu, m, beta, max_iter)
