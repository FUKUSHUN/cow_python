import os, sys
import datetime
import numpy as np
import pandas as pd
import statistics # 標準偏差の算出
import networkx as nx # Louvain法
import community # Louvain法
import matplotlib.pyplot as plt # ネットワークグラフ描画
import pdb

class BehaviorLoader:
    cow_id: int
    data: pd.DataFrame # (Time, Velocity, Behavior)

    def __init__(self, cow_id, date:datetime.datetime):
        self.cow_id = cow_id
        self._load_file(date)

    def _load_file(self, date:datetime.datetime):
        column_names = ['Time', 'Velocity', 'Behavior']
        filename = "behavior_classification/test/" + date.strftime("%Y%m%d/") + str(self.cow_id) + ".csv"
        df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,1,2,3], names=['index']+column_names,index_col=0) # csv読み込み
        
        # --- 1秒ごとのデータに整形し、self.dataに登録 ---
        revised_data = [] # 新規登録用リスト
        before_t = datetime.datetime.strptime(date.strftime("%Y%m%d"), "%Y%m%d") + datetime.timedelta(hours=12) # 正午12時を始まりとする
        before_v = 0.0
        before_b = 0 # 休息
        for _, row in df.iterrows():
            t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") # datetime
            v = float(row[1])
            b = int(row[2])
            # --- 1秒ずつのデータに直し、データごとのずれを補正する ---
            while (before_t < t):
                row = (before_t, before_v, before_b)
                revised_data.append(row)
                before_t += datetime.timedelta(seconds=1)
            before_t = t
            before_v = v
            before_b = b
        end_t = datetime.datetime.strptime(date.strftime("%Y%m%d"), "%Y%m%d") + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌日午前9時を終わりとする
        while (before_t < end_t):
            t = before_t
            if (before_t < t + datetime.timedelta(seconds=5)):
                row = (before_t, before_v, before_b) # 最後のtをまだ登録していないので登録する
            else:
                row = (before_t, 0.0, 0) # 残りの時間は休息，速度ゼロで埋める
            revised_data.append(row)
            before_t += datetime.timedelta(seconds=1)
        self.data = pd.DataFrame(revised_data, columns=column_names)
    
    def get_data(self):
        return self.data

class Synchronizer:
    date: datetime.datetime
    cow_id_list: list
    cow_id_combination_list: list # 2頭の牛の組み合わせ（ID）をリスト化して保持（何度も用いるためリスト化して保持しておく）
    cow_behavior_dict: dict # IDをキーにして牛の行動のdfを保持
    score_dict: dict # 行動同期スコアを格納, cow_id_combination_listのインデックスをキーにする

    def __init__(self, date, cow_id_list):
        self.date = date
        cow_id_list = self._confirm_csv(cow_id_list)
        self.cow_id_list = cow_id_list
        self._prepare()

    def _confirm_csv(self, cow_id_list):
        """ 行動分類のファイルが存在しているか確認しなければIDのリストから削除する """
        dir_path = "behavior_classification/test/" + self.date.strftime("%Y%m%d/")
        delete_list = []
        for cow_id in cow_id_list:
            filepath = dir_path + str(cow_id) + ".csv"
            if (not os.path.isfile(filepath)):
                cow_id_list.remove(cow_id)
                delete_list.append(cow_id)
        print("行動分類ファイルの存在しない牛のIDをリストから削除しました. 削除した牛のリスト: ", delete_list)
        return cow_id_list
    
    def _prepare(self):
        """ インタラクショングラフ作成のための準備を行う
            1. 牛の行動分類データ（1日分）を全頭分作成する
            2. 2頭の組み合わせについて数え上げる """
        print("行動分類データを読み込んでいます...")
        self.cow_behavior_dict = {} # 牛のIDでdfを管理
        for cow_id in self.cow_id_list:
            loader = BehaviorLoader(cow_id, self.date)
            df = loader.get_data()
            self.cow_behavior_dict[cow_id] = df

        self.cow_id_combination_list = [] # すでに調べた牛の組を格納
        for c_i in self.cow_id_list:
            for c_j in self.cow_id_list:
                cow_combi = sorted([c_i, c_j])
                if (c_i != c_j and not cow_combi in self.cow_id_combination_list):
                    self.cow_id_combination_list.append(cow_combi)
        print("正常に終了しました!")
        return

    def make_interaction_graph(self, start:datetime.datetime, interval:int):
        """ インタラクショングラフを作成する """
        self.score_dict = {}
        end = start + datetime.timedelta(minutes=interval)
        # すべての牛の組み合わせに対してスコアを算出する
        for i, combi in enumerate(self.cow_id_combination_list):
            cow_id1 = int(combi[0])
            cow_id2 = int(combi[1])
            score = self._calculate_score(cow_id1, cow_id2, start, end)
            self.score_dict[i] = score
        # グラフのエッジを結ぶ閾値を決定する
        threshold = self._determine_boundary()
        #重みなし無向グラフを作成する
        g, communities = self._make_undirected_graph(threshold)
        print("コミュニティを生成しました. ", start)
        print(communities)
        self._visualize_graph(g, communities, start) # グラフ描画
        return communities

    def _calculate_score(self, cow_id1, cow_id2, start:datetime.datetime, end:datetime.datetime):
        """ 行動同期スコアを計算する """
        delta = 5 # データ抽出間隔．単位は秒 (というよりはデータ数を等間隔でスライスしている)
        score_matrix = np.array([[1,0,0], [0,3,0], [0,0,9]])
        cow1_df = self._extract_df(self.cow_behavior_dict[str(cow_id1)], start, end, delta)
        cow2_df = self._extract_df(self.cow_behavior_dict[str(cow_id2)], start, end, delta)
        # --- 行動同期スコアの計算（論文参照） --- 
        score = 0
        for i,j in zip(cow1_df['Behavior'], cow2_df['Behavior']):
            score += score_matrix[i,j]
        return score

    def _extract_df(self, df:pd.DataFrame, start:datetime.datetime, end:datetime.datetime, delta: int):
        """ 特定の時間のデータを抽出する """
        df2 = df[(start < df["Time"]) & (df["Time"] < end)] # 抽出するときは代わりの変数を用意すること
        return df2[::delta] # 等間隔で抽出する
    
    def _determine_boundary(self):
        """ インタラクショングラフのエッジの有無を決定する境界線を決定する """
        scores = list(self.score_dict.values())
        X = sorted(scores)
        X1, X2 = X[0:2], X[2:]
        min_d, min_i = None, 2
        # ソートした数値を小さい方から分岐していき重心間距離の2乗の総和が最小となる分岐点を (ほぼ) 全数探索する
        for i in range(2, len(X)-2):
            x1_g = sum(X1) / len(X1) # クラスタ1の重心
            x2_g = sum(X2) / len(X2) # クラスタ2の重心
            d = 0
            for x in X1:
                d += (x - x1_g) ** 2
            for x in X2:
                d += (x - x2_g) ** 2
            # 最小値の更新
            if (min_d is None or d < min_d):
                min_d = d
                min_i = i
            X1, X2 = X[0:i+1], X[i+1:]
        threshold = (X[min_i-1] + X[min_i]) / 2 # 分割点をエッジを結ぶ閾値とする
        return threshold

    def _make_undirected_graph(self, threshold):
        """ 重みなし無向グラフを作成する """
        g = nx.Graph()
        edges = []
        # ノードメンバを登録
        for cow_id in self.cow_id_list:
            g.add_node(cow_id)
        # エッジを追加
        edges = [self.cow_id_combination_list[i] for i in range(len(self.cow_id_combination_list)) if threshold <= self.score_dict[i]]
        g.add_edges_from(edges)
        partition = community.best_partition(g)
        return g, self._make_node_list(partition)

    def _make_node_list(self, partition):
        """ コミュニティごとに各牛の個体番号のリストを作成し，コミュニティのリストを返す  """
        nodes_list = []
        for com in set(partition.values()):
            nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            nodes_list.append(nodes)
        return nodes_list

    def _visualize_graph(self, g, communities:list, date:datetime.datetime):
        """ インタラクショングラフを描画する """
        save_path = "./behavior_synchronization/graph/" + date.strftime("%Y%m%d/")
        num_nodes = len(g.nodes)
        node_colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (0,0,0), (0,0.5,0.5), (0.5,0,0.5),(0.5,0.5,0)]
        color_list = []
        for cow_id in self.cow_id_list:
            for i, com in enumerate(communities):
                if (cow_id in com):
                    color_list.append(node_colors[i%len(node_colors)])

        plt.figure(figsize=(10, 8))
        # 円状にノードを配置
        pos = {
            n: (np.cos(2*i*np.pi/num_nodes), np.sin(2*i*np.pi/num_nodes))
            for i, n in enumerate(g.nodes)
        }
        #ノードとエッジの描画
        nx.draw_networkx_edges(g, pos, edge_color='y')
        nx.draw_networkx_nodes(g, pos, node_color=color_list, alpha=0.5) # alpha: 透明度の指定
        nx.draw_networkx_labels(g, pos, font_size=10) #ノード名を付加
        plt.axis('off') #X軸Y軸を表示しない設定
        self._confirm_dir(save_path)
        plt.savefig(save_path + date.strftime("%H%M.jpg"))
        return

    def _confirm_dir(self, dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return