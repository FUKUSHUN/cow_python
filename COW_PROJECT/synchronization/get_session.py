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
start = datetime.datetime(2018, 10, 1, 0, 0, 0)
end = datetime.datetime(2018, 10, 30, 0, 0, 0)
target_list = ['20113','20170','20295','20299']
cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
change_point_file = "./synchronization/change_point/"
corpus_file = "./synchronization/topic_model/corpus/"
corpus = []

def create_corpus():
    """ コーパスを作成し，ファイルに書き込む """
    date = start
    while (date < end):
        s1 = time.time()
        communities_list = []
        community_graph_list = []
        interaction_graph_list = []
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        cow_id_list = com_creater.cow_id_list
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, delta=delta_s, leng=leng) \
                if (t_start <= t) else [[]] # コミュニティを決定
            com_creater.visualize_position(t, t+datetime.timedelta(minutes=delta_c), community, target_cow_id='20170', delta=delta_s) # 位置情報とコミュニティをプロット1
            community_graph = com_creater.get_community_graph(community)
            interaction_graph_list.append(interaction_graph)
            communities_list.append(community)
            community_graph_list.append(community_graph)
            t += datetime.timedelta(minutes=delta_c)
        e1 = time.time()
        print("処理時間", (e1-s1)/60, "[min]")
        # --- 変化点を検知し，セッションを作る ---
        s2 = time.time()
        behavior_synch = com_creater.get_behavior_synch()
        # set_analyzer = SetSeriesAnalysis(cow_id_list, communities_list)
        graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list, "Poisson")
        for cow_id in target_list:
            if (cow_id in cow_id_list):
                # change_points = set_analyzer.detect_change_point(cow_id, t_list)
                graph_analyzer.visualize_graph(cow_id, t_list) # グラフをまとめて可視化
                change_points, score_list = graph_analyzer.detect_change_point(cow_id, 5, 5) # 変化点検知
                df = pd.concat([pd.Series(t_list), pd.Series(score_list), pd.Series(change_points)], axis=1, names=["time", "score", "change_flag"])
                df.to_csv("./synchronization/graph_operation/"+ str(cow_id) + ".csv") # csvで保存
                community_list = make_session.get_focused_community(communities_list, cow_id) # セッションを作成するために対象牛の所属するコミュニティを抽出
                cow_id_session = make_session.process_time_series(t_list, community_list, change_points) # 牛IDでセッションを作成
                space_session = make_session.exchange_cowid_to_space(cow_id_session, behavior_synch, delta_c, delta_s) # 特徴表現でセッションを作成
                corpus.extend(space_session)
                session_io.write_session(space_session, corpus_file+cow_id + "/" + date.strftime("%Y%m%d/"))
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        date += datetime.timedelta(days=1)
        pdb.set_trace()
    return

def load_corpus():
    """ コーパスをファイルからロードする """
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

def predict_session(gaussian_lda, theta):
    """ 新にセッションを作り，インタラクション列を追加する """
    date = start
    while (date < end):
        s1 = time.time()
        communities_list = []
        interaction_graph_list = []
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        cow_id_list = com_creater.cow_id_list
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
            community_list = make_session.get_focused_community(communities_list, cow_id)
            cow_id_session = make_session.process_time_series(t_list, community_list, change_points)
            space_session = make_session.exchange_cowid_to_space(cow_id_session, behavior_synch, delta_c, delta_s, dim=2)
            topic_dist = gaussian_lda.predict(space_session, theta) # 予測結果．各トピックの分布で現れる
            result = []
            for p in topic_dist:
                result.append(np.where(p==max(p))[0][0]) # 最も予測確率が高いものをトピックに据える
            time_series_result = make_session.restore_time_series(t_list, change_points, [topic_dist, result]) # 結果を時系列データに直す
            df = pd.concat([df, pd.DataFrame(time_series_result[0]), pd.Series(time_series_result[1])], axis=1)
            session_io._confirm_dir(change_point_file+str(cow_id)) # ディレクトリのパスを作る
            df.to_csv(change_point_file+str(cow_id)+date.strftime("/%Y%m%d.csv"))
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        date += datetime.timedelta(days=1)
    return

if __name__ == "__main__":
    is_create = True
    is_load = False
    is_learn = False
    if (is_create):
        corpus = []
        create_corpus()
    if (is_load):
        corpus = []
        load_corpus()
    if (is_learn):
        # LDAのハイパーパラメータ設定
        M = len(corpus)
        K = 5
        alpha = np.array([[1, 1, 1, 1, 1] for _ in range(M)]) # parameter for dirichlet
        psi = np.array([[[3.73512784e-07, 1.69779886e-08], [1.69779886e-08, 2.85138486e-08]], \
                    [[5.05955252e-06, 5.67102212e-06], [5.67102212e-06, 6.96904853e-06]], \
                        [[3.42327177e-07, 3.24412304e-07], [3.24412304e-07, 3.75268994e-07]], \
                            [[2.58528836e-07, 1.90666739e-07], [1.90666739e-07, 2.80431611e-07]], \
                                [[4.90882946e-07, 4.63907573e-07], [4.63907573e-07, 6.17471018e-07]]]) # parameter for Gaussian Wishert
        m = np.array([[0.0168134, 0.49485071], [0.97943516, 0.01868553], [0.28869599, 0.59634904], [0.29941286, 0.39345673], [0.67838774, 0.241999]]) # parameter for Gaussian Wishert
        nu = np.array([6.85110703e+08, 1.05511992e+09, 2.35658659e+08, 1.41584940e+08, 2.30210528e+08]) # parameter for Gaussian Wishert
        beta = np.array([6.85110703e+08, 1.05511992e+09, 2.35658659e+08, 1.41584940e+08, 2.30210528e+08]) # parameter for Gaussian Wishert
        max_iter = 1000

        # ギブスサンプリングによるクラスタリング
        gaussian_lda = GaussianLDA(corpus = corpus, num_topic=5, dimensionality=2)
        Z, theta = gaussian_lda.inference(alpha, psi, nu, m, beta, max_iter)
        result = gaussian_lda.predict(corpus, theta)
    else:
        beta = np.array([1.05669156e+09, 2.13276228e+08, 2.14682579e+08, 7.26673544e+08, 1.36360834e+08])
        m = np.array([[0.97754441, 0.01987148], [0.29367018, 0.56291046], [0.67696687, 0.24723722], [0.02296877, 0.50454176], [0.39541451, 0.36614583]])
        nu = np.array([1.05669156e+09, 2.13276228e+08, 2.14682579e+08, 7.26673544e+08, 1.36360834e+08])
        W = np.array([[[3.12971170e-06, 3.65023628e-06], [3.65023628e-06, 4.77065779e-06]],\
                        [[2.63659347e-07, 2.35619748e-07], [2.35619748e-07, 2.87111953e-07]], \
                            [[5.15136665e-07, 5.09762336e-07], [5.09762336e-07, 6.83040306e-07]], \
                                [[2.34470246e-07, 1.39159975e-08], [1.39159975e-08, 2.54139520e-08]], \
                                    [[2.38485200e-07, 2.06301212e-07], [2.06301212e-07, 3.24740050e-07]]])
        theta = np.array([0.450009512 , 0.09084722, 0.09144625, 0.30952585, 0.05808556])
        gaussian_lda = GaussianLDA(corpus = corpus, num_topic=5, dimensionality=2)
        gaussian_lda.set_params(beta, m, nu, W)
        result = gaussian_lda.predict(corpus, theta)
    predict_session(gaussian_lda, theta)
    
