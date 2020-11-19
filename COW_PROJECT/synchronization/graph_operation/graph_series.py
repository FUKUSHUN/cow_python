import time
import datetime
import sys, os
import numpy as np
import pandas as pd
import pdb

# my class
from synchronization.graph_operation.graph import GraphAnalysis
# from graph import GraphAnalysis

class GraphSeriesAnalysis:
    """ グラフの系列（時系列）を扱い、変化点を検出する機能を備えるクラス """
    cow_id_list: list # グラフのインデックスに対応する牛の個体番号を表すリスト
    graph_series: list # グラフ構造を表す隣接行列のリスト
    graph_type: str # 各エッジが従う確率分布
 
    def __init__(self, cow_id_list, graph_series, graph_type):
        self.cow_id_list = cow_id_list
        self.graph_series = graph_series
        self.graph_type = graph_type

    def detect_change_point(self, target_cow_id, W1, W2, threshold=10000):
        """ 変化点をグラフの変化から検知する（外部から呼び出されるメインのメソッド）
            target_cow_id: 変化点検知の対象とする牛の個体番号
            W1: int Window size for comparison (before current time)
            W2: int Window size for comparison (after current time) """
        changepoint_list = [] # 変化点なら1, そうでなければ0
        score_list = [] # # 非類似度スコアを格納
        target_cow_index = self.cow_id_list.index(str(target_cow_id))
        reset_count = 0
        for i in range(len(self.graph_series)):
            if (i + W2 > len(self.graph_series)): # 現在時刻から後ろにW2個のグラフがない場合はそこで変化点検知を終了する
                break
            if (reset_count < W1): # 変化点から最初のW1個のグラフは変化点ではないとする
                changepoint_list.append(0) # 変化点なら1, そうでなければ0
                score_list.append(0) # 非類似度スコア
                reset_count += 1
            else: # コミュニティの変化点検知を行う
                before_graphs = self.graph_series[i - W1 : i] # 現在時刻より前W1個のグラフ
                after_graphs = self.graph_series[i : i + W2] # 現在時刻から後W2個のグラフ
                score = self._compare_graph(target_cow_index, before_graphs, after_graphs)
                # 変化点となる時刻を格納する
                if(threshold < score):
                    changepoint_list.append(1)# 変化点なら1, そうでなければ0
                    score_list.append(score) # 非類似度スコア
                    reset_count = 0 # カウントリセット
                else:
                    changepoint_list.append(0)# 変化点なら1, そうでなければ0
                    score_list.append(score) # 非類似度スコア
        return changepoint_list, score_list
    
    def visualize_graph(self, target_cow_id, t_list):
        """ グラフの可視化を行う """
        save_path = "./visualization/graph/" + str(target_cow_id) + t_list[0].strftime("/%Y%m%d/")
        target_cow_index = self.cow_id_list.index(str(target_cow_id))
        self._confirm_dir(save_path)
        for i, graph in enumerate(self.graph_series):
            filename = t_list[i].strftime("%H%M.png")
            ga = GraphAnalysis(None, None, None)
            ga.visualize_graph(graph, target_cow_index, self.cow_id_list, save_path, filename)
        return

    def _compare_graph(self, target_index, graphs1, graphs2):
        """ 2つのグラフ集合を比較し、非類似度を返す
            target_index: 変化点検知の対象となる牛の隣接行列内の番号（何行（何列）目か）
            graphs1, graphs2: list(np.array(2d))    グラフ（隣接行列）のリスト """
        ga = GraphAnalysis(graphs1, graphs2, self.graph_type) # graph1が現在，graph2が直前
        score = ga.measure_similarity(target_index) # 2つのグラフの構造的類似度
        return score

    def _confirm_dir(self, dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return

if __name__ == "__main__":
    os.chdir('../../') # カレントディレクトリを一階層上へ
    print(os.getcwd())
    sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
    import synchronization.community_creater as community_creater
    import synchronization.functions.utility as my_utility
    from synchronization.graph_operation.graph_series import GraphSeriesAnalysis
    from synchronization.graph_operation.graph import GraphAnalysis
    delta_c = 2 # コミュニティの抽出間隔 [minutes]
    delta_s = 5 # データのスライス間隔 [seconds] 
    epsilon = 12 # コミュニティ決定のパラメータ
    dzeta = 12 # コミュニティ決定のパラメータ
    leng = 5 # コミュニティ決定のパラメータ
    date = datetime.datetime(2018, 10, 21, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    output_file = "./synchronization/test/graph_similarity/"
    target_list = ['20113','20170','20295','20299']
    s1 = time.time()
    t_list = []
    interaction_graph_list = []
    community_graph_list = []
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
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
        community_graph_list.append(community_graph)
        t += datetime.timedelta(minutes=delta_c)
    e1 = time.time()
    print("処理時間", (e1-s1)/60, "[min]")
    # --- 次のコミュニティを予測する ---
    s2 = time.time()
    graph_analyzer = GraphSeriesAnalysis(cow_id_list, community_graph_list, "Bernoulli")
    for cow_id in target_list:
        change_points, score_list = graph_analyzer.detect_change_point(cow_id, 5, 5)
        df = pd.concat([pd.Series(t_list), pd.Series(score_list), pd.Series(change_points)], axis=1)
        df.to_csv(output_file+cow_id+".csv")
        pdb.set_trace()
    e2 = time.time()
    print("処理時間", (e2-s2)/60, "[min]")