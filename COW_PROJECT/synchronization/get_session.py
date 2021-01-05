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
leng = 1 # コミュニティ決定のパラメータ
target_list = ['20122','20129','20158','20170','20192','20197','20215','20267','20283']
cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
change_point_file = "./synchronization/change_point/"
corpus_file = "./synchronization/topic_model/corpus/"

def create_corpus(repeat_num = 100):
    """ コーパスを作成し，ファイルに書き込む """
    corpus = []
    start = datetime.datetime(2018, 6, 1, 0, 0, 0)
    end = datetime.datetime(2018, 12, 31, 0, 0, 0)
    log_file = "./synchronization/topic_model/log_for_making_session.txt"
    my_utility.delete_file(log_file) # 一度既存のログファイルをクリーンする
    for cow_id in target_list:
        my_utility.write_logs(log_file, [str(cow_id)])
        for i in range(repeat_num):
            date = start + datetime.timedelta(days=np.random.randint(0, 200)) # startからランダムな日数を経過させる
            if (date < end): # 期間内であれば
                s1 = time.time()
                communities_list = []
                community_graph_list = []
                interaction_graph_list = []
                t_list = []
                cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
                com_creater = community_creater.CommunityCreater(date, cow_id_list)
                cow_id_list = com_creater.cow_id_list
                if (cow_id in cow_id_list): # 対象牛の位置情報があれば
                    # --- コミュニティ構造を分析する ---
                    t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
                    t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
                    t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
                    while (t < t_end):
                        t_list.append(t)
                        interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                            if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
                        community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, delta=delta_s, leng=leng) \
                            if (t_start <= t) else [[]] # コミュニティを決定
                        # com_creater.visualize_position(t, t+datetime.timedelta(minutes=delta_c), community, target_cow_id='20170', delta=delta_s) # 位置情報とコミュニティをプロット1
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
                    graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list, "Poisson")
                    # graph_analyzer.visualize_graph(cow_id, t_list) # グラフをまとめて可視化
                    change_points, score_list = graph_analyzer.detect_change_point(cow_id, 5, 5, threshold=400) # 変化点検知
                    # df = pd.concat([pd.Series(t_list), pd.Series(score_list), pd.Series(change_points)], axis=1, names=["time", "score", "change_flag"])
                    # df.to_csv("./synchronization/graph_operation/"+ str(cow_id) + ".csv") # csvで保存
                    community_list = make_session.get_focused_community(communities_list, cow_id) # セッションを作成するために対象牛の所属するコミュニティを抽出
                    cow_id_session = make_session.process_time_series(t_list, community_list, change_points) # 牛IDでセッションを作成
                    space_session = make_session.exchange_cowid_to_space(cow_id_session, behavior_synch, delta_c, delta_s) # 特徴表現でセッションを作成
                    corpus.extend(space_session)
                    session_io.write_session(space_session, corpus_file+cow_id + "/" + date.strftime("%Y%m%d/"))
                    my_utility.write_logs(log_file, [str(i+1) + "番目の試行 Date: " + date.strftime("%Y/%m/%d") + " Success"])
                    e2 = time.time()
                    print("処理時間", (e2-s2)/60, "[min]")
                else:
                    my_utility.write_logs(log_file, [str(i+1) + "番目の試行 Date: " + date.strftime("%Y/%m/%d") + " Failure"])
    return

def load_corpus():
    """ コーパスをファイルからロードする """
    log_file = "./synchronization/topic_model/log_for_loading_session.txt"
    my_utility.delete_file(log_file) # 一度既存のログファイルをクリーンする
    s1 = time.time()
    corpus = []
    # --- コーパスをファイルからロードする ---
    for cow_id in target_list:
        if (os.path.exists(corpus_file + str(cow_id))): # ディレクトリのパスが存在すれば
            my_utility.write_logs(log_file, [str(cow_id)])
            dirs = os.listdir(corpus_file + str(cow_id)) # 直下のディレクトリ（YYYYmmdd）をすべて取り出す
            for dir_name in dirs:
                date = datetime.datetime.strptime(dir_name, "%Y%m%d") # 変な名前のディレクトリが入っているとエラー出る可能性あり
                space_session = session_io.read_session(corpus_file+cow_id + "/" + date.strftime("%Y%m%d/"))
                corpus.extend(space_session)
                my_utility.write_logs(log_file, ["Date: " + date.strftime("%Y/%m/%d") + "   Success"])
    e1 = time.time()
    print("処理時間", (e1-s1)/60, "[min]")
    return corpus

def predict_session(gaussian_lda, theta):
    """ 新にセッションを作り，インタラクション列を追加する """
    start = datetime.datetime(2018, 5, 16, 0, 0, 0)
    end = datetime.datetime(2018, 7, 31, 0, 0, 0)
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
            com_creater.visualize_position(t, t+datetime.timedelta(minutes=delta_c), community, target_cow_id='20113', delta=delta_s) # 位置情報とコミュニティをプロット1
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
        for cow_id in target_list:
            if (cow_id in cow_id_list):
                graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list, "Poisson")
                # graph_analyzer.visualize_graph(cow_id, t_list) # グラフをまとめて可視化
                change_points, score_list = graph_analyzer.detect_change_point(cow_id, 5, 5, threshold=400) # 変化点検知
                df = pd.concat([pd.Series(t_list), pd.Series(score_list), pd.Series(change_points)], axis=1, names=["time", "score", "change_flag"])
                # df.to_csv("./synchronization/graph_operation/"+ str(cow_id) + ".csv") # csvで保存
                community_list = make_session.get_focused_community(communities_list, cow_id) # セッションを作成するために対象牛の所属するコミュニティを抽出
                cow_id_session = make_session.process_time_series(t_list, community_list, change_points) # 牛IDでセッションを作成
                space_session = make_session.exchange_cowid_to_space(cow_id_session, behavior_synch, delta_c, delta_s, dim=2) # 特徴表現でセッションを作成
                topic_dist = gaussian_lda.predict(space_session, theta) # 予測結果．各トピックの分布で現れる
                result = [np.where(p==max(p))[0][0] for p in topic_dist] # 最も予測確率が高いものをトピックに据える
                time_series_result = make_session.restore_time_series(t_list, change_points, [topic_dist, result]) # 結果を時系列データに直す
                df = pd.concat([df, pd.DataFrame(time_series_result[0]), pd.Series(time_series_result[1])], axis=1)
                my_utility.confirm_dir(change_point_file+str(cow_id)) # ディレクトリのパスを作る
                df.to_csv(change_point_file+str(cow_id)+date.strftime("/%Y%m%d.csv"))
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        date += datetime.timedelta(days=1)
    return

if __name__ == "__main__":
    is_create = False
    is_learn = False
    corpus = []
    
    # --- セッションを新規作成する ---
    if (is_create):
        for _ in range(10):
            create_corpus(repeat_num = 10) # 10回繰り返しを10回繰り返す
        pdb.set_trace()
    
    # --- コーパスを読み込む ---
    corpus = load_corpus()
    corpus = np.random.choice(corpus, int(np.round(0.7 * len(corpus))), replace=False) # corpusの中から間引いて学習に使用する
    # pdb.set_trace()

    # --- セッションの学習 or パラメータ割り当て ---
    if (is_learn): # パラメータ学習
        # LDAのハイパーパラメータ設定
        M = len(corpus)
        K = 5
        alpha = np.array([[1, 1, 1, 1] for _ in range(M)]) # parameter for dirichlet
        psi = np.array([[[2.0, 0.0], [0.0, 2.0]], \
                    [[2.0, 0.0], [0.0, 2.0]], \
                        [[2.0, 0.0], [0.0, 2.0]], \
                            [[2.0, 0.0], [0.0, 2.0]]]) # parameter for Gaussian Wishert
        m = np.array([[1, 0], [0, 1], [0, 0], [0.3, 0.3]]) # parameter for Gaussian Wishert
        nu = np.array([5, 5, 5, 5]) # parameter for Gaussian Wishert
        beta = np.array([5.0, 5.0, 5.0, 5.0]) # parameter for Gaussian Wishert
        max_iter = 1500

        # ギブスサンプリングによるクラスタリング
        gaussian_lda = GaussianLDA(corpus = corpus, num_topic=4, dimensionality=2)
        Z, theta = gaussian_lda.inference(alpha, psi, nu, m, beta, max_iter)
        # result = gaussian_lda.predict(corpus, theta)
    else: # 学習しない (定数値を与える)
        beta = np.array([589307163.460198, 323020503.759785, 72738742.85204238, 330578121.927975])
        m = np.array([[0.9806218714269278, 0.01927556323524467], [0.00046157409354147483, 0.5990943621538058], [0.17536538609218583, 0.35482909625335546], [0.5780805567607266, 0.3189921501692251]])
        nu = np.array([589307163.460198, 323020503.759785, 72738742.85204238, 330578121.927975])
        W = np.array([[[0.00023267719356337007, 0.00023331524100325075], [0.00023331524100325086, 0.000235484452004462]], \
            [[9.30688634377732e-05, -6.9804035862662424e-09], [-6.980403586266246e-09, 5.273889183190476e-08]], \
                [[9.080984813633261e-07, 8.13808890425863e-08], [8.138088904258649e-08, 3.6969362276004587e-07]], \
                    [[2.3299351405514757e-07, 2.236769622813933e-07], [2.2367696228139326e-07, 2.792943056476546e-07]]])
        theta = np.array([0.450009512 , 0.09084722, 0.09144625, 0.30952585, 0.05808556])
        gaussian_lda = GaussianLDA(corpus = corpus, num_topic=4, dimensionality=2)
        gaussian_lda.set_params(beta, m, nu, W)
        # result = gaussian_lda.predict(corpus, None)
    
    # --- 新しいセッションを予測する ---
    predict_session(gaussian_lda, None)
    
