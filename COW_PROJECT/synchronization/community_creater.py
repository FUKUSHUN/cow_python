import os, sys
import datetime
import pandas as pd
import numpy as np
import pdb
import networkx as nx # ネットワークグラフ
import community # Louvain法
import matplotlib.pyplot as plt # ネットワークグラフ描画

# 自作メソッド
import cows.geography as geography
# 自作クラス
import behavior_information.synchronizer as behavior_synchronizer # 行動同期
import position_information.synchronizer as position_synchronizer # 空間同期
import visualization.place_plot as place_plot

class CommunityCreater:
    date: datetime.datetime
    cow_id_list: list
    cow_id_combination_list: list # 2頭の牛の組み合わせ（ID）をリスト化して保持（何度も用いるためリスト化して保持しておく）
    behavior_synch: behavior_synchronizer.Synchronizer # 自作クラス, pd.DataFrame形式のデータを保持
    position_synch: position_synchronizer.Synchronizer # 自作クラス, pd.DataFrame形式のデータを保持
    score_dict: dict # 行動同期スコアを格納, cow_id_combination_listのインデックスをキーにする

    def __init__(self, date, cow_id_list):
        self.date = date
        self.behavior_synch = behavior_synchronizer.Synchronizer(date, cow_id_list)
        self.position_synch = position_synchronizer.Synchronizer(date, cow_id_list)
        self.cow_id_list = self._check_cow_id_list()
        self._make_combination_list()
        return

    def _check_cow_id_list(self):
        cl1 = self.behavior_synch.get_cow_id_list()
        cl2 = self.position_synch.get_cow_id_list()
        return sorted(list(set(cl1) & set(cl2))) # 二つの牛のIDリストの重複を牛のリスト, 必ずソートする

    def _make_combination_list(self):
        """ 2頭の組み合わせについて数え上げる """
        self.cow_id_combination_list = [] # すでに調べた牛の組を格納
        for c_i in self.cow_id_list:
            for c_j in self.cow_id_list:
                cow_combi = sorted([c_i, c_j])
                if (c_i != c_j and not cow_combi in self.cow_id_combination_list):
                    self.cow_id_combination_list.append(cow_combi)
        return

    def make_interaction_graph(self, start:datetime.datetime, interval:int, method="position"):
        """ インタラクショングラフを作成する, methodは複数用意する予定 """
        delta = 5 # データ抽出間隔．単位は秒 (というよりはデータ数を等間隔でスライスしている)
        epsilon = 30 # 距離の閾値．単位はメートル（行動同期を見る際にも距離により明らかな誤認識を避ける）
        dzeta = 20 # 距離の閾値. 単位はメートル（空間同期を見る際に基準となる閾値）
        self.score_dict = {}
        end = start + datetime.timedelta(minutes=interval)
        # すべての牛の組み合わせに対してスコアを算出する
        df = self._extract_and_merge_df(start, end, delta=delta) # データを抽出し結合
        for i, combi in enumerate(self.cow_id_combination_list):
            cow_id1 = int(combi[0])
            cow_id2 = int(combi[1])
            # インタラクションの決定法によって処理を分岐する
            if (method == "behavior"):
                score = self._calculate_behavior_synchronization(df, cow_id1, cow_id2, start, end, epsilon=epsilon)
            elif (method == "position"):
                score = self._calculate_position_synchronization(df, cow_id1, cow_id2, start, end, dzeta=dzeta)
            self.score_dict[i] = score
        # グラフのエッジを結ぶ閾値を決定する
        threshold = self._determine_boundary()
        #重みなし無向グラフを作成する
        g, communities = self._make_undirected_graph(threshold)
        print("コミュニティを生成しました. ", start)
        print(communities) 
        self._visualize_graph(g, communities, start) # グラフ描画
        pos_df = self.position_synch.extract_df(start, end, delta)
        self._visualize_community(pos_df, communities) # 動画描画
        return communities

    def _extract_and_merge_df(self, start, end, delta=5):
        """ startからendまでの時間のデータをdeltaごとにスライスして抽出し，行動，空間の2つのデータを結合する(どちらも1秒ごとに成形し，インデックスがTimeになっている前提)
            delta   : int. 単位は[s (個)]. この個数ごとに等間隔でデータをスライス """
        beh_df = self.behavior_synch.extract_df(start, end, delta)
        pos_df = self.position_synch.extract_df(start, end, delta)
        merged_df = pd.concat([beh_df, pos_df], axis=1)
        return merged_df

    def _calculate_behavior_synchronization(self, df, cow_id1, cow_id2, start:datetime.datetime, end:datetime.datetime, epsilon=30):
        """ 行動同期スコアを計算する
            epsilon : int. 単位は [m]. この距離以内の時行動同期を測定する（この距離以上のとき同期していても0）． """
        score_matrix = np.array([[1,0,0], [0,3,0], [0,0,9]])
        df2 = df[[str(cow_id1), str(cow_id2)]] # 2頭を抽出
        # --- 行動同期スコアの計算（論文参照） --- 
        score = 0
        for _, row in df2.iterrows():
            lat1, lon1, lat2, lon2 = row[1][0], row[1][1], row[3][0], row[3][1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            # 距離が閾値以内ならスコアを加算する
            if (dis <= epsilon):
                score += score_matrix[row[0][1],row[2][1]]
        return score

    def _calculate_position_synchronization(self, df, cow_id1, cow_id2, start:datetime.datetime, end:datetime.datetime, dzeta=10):
        """ 空間同期スコアを計算する 
            dzeta   : int. 単位は [m]. この距離以内の時間を計測する """
        df2 = df[[str(cow_id1), str(cow_id2)]] # 2頭を抽出
        # --- 行動同期スコアの計算（論文参照） --- 
        score = 0
        for _, row in df2.iterrows():
            lat1, lon1, lat2, lon2 = row[1][0], row[1][1], row[3][0], row[3][1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            if (dis <= dzeta):
                score += 1
        return score

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
        save_path = "./synchronization/graph/" + date.strftime("%Y%m%d/")
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

    def _visualize_community(self, df, communities:list):
        caption_list = []
        color_list = []
        for cow_id in self.cow_id_list:
            for i, com in enumerate(communities):
                if (cow_id in com):
                    caption_list.append(str(cow_id) + ":" + str(i))
                    color_list.append(i)
                    break
        maker = place_plot.PlotMaker(caption_list=caption_list, color_list=color_list)
        maker.make_movie(df, disp_adj=False)

    def _confirm_dir(self, dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return