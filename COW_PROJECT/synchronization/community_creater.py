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
import synchronization.clustering.dbscan as dbscan
import synchronization.clustering.louvain as louvain

class CommunityCreater:
    date: datetime.datetime
    cow_id_list: list
    behavior_synch: behavior_synchronizer.Synchronizer # 自作クラス, pd.DataFrame形式のデータを保持
    position_synch: position_synchronizer.Synchronizer # 自作クラス, pd.DataFrame形式のデータを保持

    def __init__(self, date, cow_id_list):
        self.date = date
        self.behavior_synch = behavior_synchronizer.Synchronizer(date, cow_id_list)
        self.position_synch = position_synchronizer.Synchronizer(date, cow_id_list)
        self.cow_id_list = self._check_cow_id_list()
        return

    def _check_cow_id_list(self):
        cl1 = self.behavior_synch.get_cow_id_list()
        cl2 = self.position_synch.get_cow_id_list()
        return sorted(list(set(cl1) & set(cl2))) # 二つの牛のIDリストの重複を牛のリスト, 必ずソートする


    def create_community(self, start:datetime.datetime, interval:int, method="position", visualized_g=False, visualized_m=False, focusing_cow_id=None):
        """ インタラクショングラフを作成する, methodは複数用意する予定
            interval: int   : startから何分間とするか
            method: str     : position or behavior, コミュニティ作成手法
            visualize_g, visualized_m: bool   : グラフ保存，動画保存をするか
            focusing_cow_id : 指定があればこの牛のいるコミュニティのみを返却する """
        delta = 5 # データ抽出間隔．単位は秒 (というよりはデータ数を等間隔でスライスしている)
        epsilon = 12 # 距離の閾値．単位はメートル（行動同期を見る際にも距離により明らかな誤認識を避ける）
        dzeta = 10 # 距離の閾値. 単位はメートル（空間同期を見る際に基準となる閾値）
        score_list = []
        end = start + datetime.timedelta(minutes=interval)
        # すべての牛の組み合わせに対してスコアを算出する
        K = len(self.cow_id_list)
        W = np.zeros([K,K])
        df, _, pos_df = self._extract_and_merge_df(start, end, delta=delta) # データを抽出し結合
        for i, cow_id1 in enumerate(self.cow_id_list):
            for j, cow_id2 in enumerate(self.cow_id_list):
                # インタラクションの決定法によって処理を分岐する
                if (i < j):
                    if (method == "behavior"):
                        score = self._calculate_behavior_synchronization(df, cow_id1, cow_id2, epsilon=epsilon)
                    elif (method == "position"):
                        score = self._calculate_position_synchronization(df, cow_id1, cow_id2, dzeta=dzeta)
                    elif (method == "distance"): # 重みと距離が反比例の関係にあるのでそのままではDBSCANにしか使えないことに注意
                        score = self._calculate_average_distance(df, self.cow_id_list[i], self.cow_id_list[j])
                    score_list.append(score)
                    W[i,j] = score
                    W[j,i] = score
                elif (i == j):
                    W[i,j] = 0
                else:
                    continue
        # グラフのエッジを結ぶ閾値を決定する
        threshold = self._determine_boundary(score_list)
        #重みなし無向グラフを作成する
        X = np.zeros([K, K])
        for i in range(K):
            for j in range(i,K):
                X[i,j] = 1 if threshold <= W[i,j] else 0
                X[j,i] = 1 if threshold <= W[j,i] else 0
        louvainer = louvain.CommunityLouvain()
        communities = louvainer.create_community(self.cow_id_list, W)
        # g = louvainer.create_graph(self.cow_id_list, X)
        g = louvainer.create_weighted_graph(self.cow_id_list, W)
        # eps, minPts = 10, 2
        # dbscanner = dbscan.CommunityDBSCAN(eps, minPts)
        # communities = dbscanner.create_community(W, self.cow_id_list)
        print("コミュニティを生成しました. ", start)
        print(communities)
        if (visualized_g):
            self._visualize_graph(g, communities, start, weighted=True) # グラフ描画
        if (visualized_m):
            self._visualize_community(pos_df, communities) # 動画描画
        if (focusing_cow_id is not None):
            community = None
            for com in communities:
                if (str(focusing_cow_id) in com):
                    community = com
                    break
            return community # 指定がある場合はその牛が所属するコミュニティのみ返す
        return communities

    def _extract_and_merge_df(self, start, end, delta=5):
        """ startからendまでの時間のデータをdeltaごとにスライスして抽出し，行動，空間の2つのデータを結合する(どちらも1秒ごとに成形し，インデックスがTimeになっている前提)
            delta   : int. 単位は[s (個)]. この個数ごとに等間隔でデータをスライス """
        beh_df = self.behavior_synch.extract_df(start, end, delta)
        pos_df = self.position_synch.extract_df(start, end, delta)
        merged_df = pd.concat([beh_df, pos_df], axis=1)
        return merged_df, beh_df, pos_df

    def _calculate_behavior_synchronization(self, df, cow_id1, cow_id2, epsilon=30):
        """ 行動同期スコアを計算する
            epsilon : int. 単位は [m]. この距離以内の時行動同期を測定する（この距離以上のとき同期していても0）． """
        score_matrix = np.array([[1,0,0], [0,1,0], [0,0,9]])
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

    def _calculate_position_synchronization(self, df, cow_id1, cow_id2, dzeta=10):
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

    def _calculate_average_distance(self, df, cow_id1, cow_id2):
        """ 距離の平均を算出する """
        df2 = df[[str(cow_id1), str(cow_id2)]] # 2頭を抽出
        # --- 行動同期スコアの計算（論文参照） --- 
        count = 0
        accumulated_distance = 0
        for _, row in df2.iterrows():
            lat1, lon1, lat2, lon2 = row[1][0], row[1][1], row[3][0], row[3][1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            # 距離を加算する
            accumulated_distance += dis
            count += 1
        return accumulated_distance / count

    def _determine_boundary(self, score_list):
        """ インタラクショングラフのエッジの有無を決定する境界線を決定する """
        X = sorted(score_list)
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

    def _visualize_graph(self, g, communities:list, date:datetime.datetime, weighted=False):
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
        if (weighted):
            edge_labels = {(i, j): w['weight'] for i, j, w in g.edges(data=True)}
            nx.draw_networkx_edge_labels(g,pos, edge_labels=edge_labels)
        plt.axis('off') #X軸Y軸を表示しない設定
        self._confirm_dir(save_path)
        plt.savefig(save_path + date.strftime("%H%M.jpg"))
        return

    def _visualize_community(self, df, communities:list):
        """ 位置情報可視化動画を作成する """
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
        return

    def _visualize_adjectory(self, df, communities, focusing_cow_id=None):
        """ 軌跡描画を行う
            focusing_cow_id: str  Noneのときは全コミュニティを描画，指定ありの場合は当該コミュニティのみを描画 """
        caption_list = []
        color_list = []
        if (focusing_cow_id is None):
            for cow_id in self.cow_id_list:
                for i, com in enumerate(communities):
                    if (cow_id in com):
                        caption_list.append("")
                        color_list.append(i)
                        break
            maker = place_plot.PlotMaker(caption_list=caption_list, color_list=color_list)
            maker.make_adjectory(df)
        else:
            community = [] # focusing_cow_idが所属するコミュニティを格納する
            for com in communities:
                if (focusing_cow_id in com):
                    community = com
                    break
            community = sorted(community)
            for cow_id in community:
                if (cow_id == focusing_cow_id):
                    caption_list.append("") # キャプションを表示しない
                    color_list.append(0)
                else:
                    caption_list.append("") # キャプションを表示しない
                    color_list.append(1)
            new_df = df[community] # communityを使ってdfから必要な要素を抽出
            maker = place_plot.PlotMaker(caption_list=caption_list, color_list=color_list, image_filename=str(focusing_cow_id)+"/")
            maker.make_adjectory(new_df)
        return

    def _confirm_dir(self, dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return

    def get_behavior_synch(self):
        """ behavior_synchはロードに時間がかかるので使いまわす """
        return self.behavior_synch

    def get_position_synch(self):
        """ position_synchはロードに時間がかかるので使いまわす """
        return self.position_synch