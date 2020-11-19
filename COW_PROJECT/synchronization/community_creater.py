import os, sys
import math
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
    community_history:list # コミュニティ履歴（要素数はlen以下になる）
    date: datetime.datetime
    cow_id_list: list
    behavior_synch: behavior_synchronizer.Synchronizer # 自作クラス, pd.DataFrame形式のデータを保持
    position_synch: position_synchronizer.Synchronizer # 自作クラス, pd.DataFrame形式のデータを保持

    def __init__(self, date, cow_id_list):
        self.community_history = []
        self.date = date
        self.behavior_synch = behavior_synchronizer.Synchronizer(date, cow_id_list)
        self.position_synch = position_synchronizer.Synchronizer(date, cow_id_list)
        self.cow_id_list = self._check_cow_id_list()
        return

    def _check_cow_id_list(self):
        cl1 = self.behavior_synch.get_cow_id_list()
        cl2 = self.position_synch.get_cow_id_list()
        return sorted(list(set(cl1) & set(cl2))) # 二つの牛のIDリストの重複を牛のリスト, 必ずソートする

    def make_interaction_graph(self, start:datetime.datetime, end:datetime.datetime, method="position", delta=5, epsilon=12, dzeta=12):
        """ 重み付きグラフのインタラクションがグラフを作成うする
            method:     str     position or behavior, コミュニティ作成手法
            delta       int     データ抽出間隔．単位は秒 (というよりはデータ数を等間隔でスライスしている)
            epsilon:    int     method="behavior"用の距離の閾値．単位はメートル
            dzeta:      int     method="position"用の距離の閾値．単位はメートル """
        beh_df, pos_df = self._extract_and_merge_df(start, end, delta=delta) # データを抽出し結合
        score_list = []
        K = len(self.cow_id_list)
        W = np.zeros([K,K]) # 重み付き行列．このメソッドの返却値
        for i, cow_id1 in enumerate(self.cow_id_list):
            for j, cow_id2 in enumerate(self.cow_id_list):
                # インタラクションの決定法によって処理を分岐する
                if (i < j):
                    if (method == "behavior"):
                        score = self._calculate_behavior_synchronization(beh_df, pos_df, cow_id1, cow_id2, epsilon=epsilon)
                    elif (method == "position"):
                        score = self._calculate_position_synchronization(pos_df, cow_id1, cow_id2, dzeta=dzeta)
                    elif (method == "distance"): # 重みと距離が反比例の関係にあるのでそのままではDBSCANにしか使えないことに注意
                        score = self._calculate_average_distance(pos_df, self.cow_id_list[i], self.cow_id_list[j])
                    score_list.append(score)
                    W[i,j] = score
                    W[j,i] = score
                elif (i == j):
                    W[i,j] = 0
                else:
                    continue
        return W

    def create_community(self, start, end, W:np.array, delta=5, leng=5):
        """ インタラクショングラフを作成する, methodは複数用意する予定
            W:          np.array            : インタラクショングラフ（重み付きグラフの行列）
            delta       int     データ抽出間隔．単位は秒 (というよりはデータ数を等間隔でスライスしている) 
            leng        int     過去の重み付きグラフの考慮数 """
        # --- 重みありグラフに対してLouvain法を適用してコミュニティを決定する（W: 重み付き）---
        louvainer = louvain.CommunityLouvain()
        G = self._calculate_dynamic_W(W, leng=leng)
        communities = louvainer.create_community(self.cow_id_list, G) # louvain法を使用してコミュニティを決定する
        g = louvainer.create_weighted_graph(self.cow_id_list, G)
        
        # #  --- 重みなしグラフに対してLouvain法を適用してコミュニティを決定する（X: 重みなし）---
        # threshold = self._determine_boundary(score_list) # グラフのエッジを結ぶ閾値を決定する
        # X = louvainer.exchange__undirected_graph(G, threshold) #重みなし無向グラフを作成する
        # communities = louvainer.create_community(self.cow_id_list, X) # louvain法を使用してコミュニティを決定する
        # g = louvainer.create_graph(self.cow_id_list, X)
        
        # #  --- DBSCANを使用してコミュニティを決定する（W: 距離行列）---
        # eps, minPts = 10, 2
        # dbscanner = dbscan.CommunityDBSCAN(eps, minPts)
        # communities = dbscanner.create_community(W, self.cow_id_list)
        print("コミュニティを生成しました. ", start)
        print(communities)
        return communities

    def visualize_position(self, start, end, communities, target_cow_id=None, delta=5):
        """ 位置情報をプロットする
            target_cow_id : 指定があればこの牛のいるコミュニティの描画の色を固定する
            delta       int     データ抽出間隔．単位は秒 (というよりはデータ数を等間隔でスライスしている)  """
        _, pos_df = self._extract_and_merge_df(start, end, delta=delta) # データを抽出し結合
        self._visualize_community(pos_df, communities, focusing_cow_id=target_cow_id) # 動画描画
        return

    def _extract_and_merge_df(self, start, end, delta=5):
        """ startからendまでの時間のデータをdeltaごとにスライスして抽出し，行動，空間の2つのデータを結合する(どちらも1秒ごとに成形し，インデックスがTimeになっている前提)
            delta   : int. 単位は[s (個)]. この個数ごとに等間隔でデータをスライス """
        beh_df = self.behavior_synch.extract_df(start, end, delta)
        pos_df = self.position_synch.extract_df(start, end, delta)
        return beh_df, pos_df

    def _calculate_behavior_synchronization(self, beh_df, pos_df, cow_id1, cow_id2, epsilon=30):
        """ 行動同期スコアを計算する
            epsilon : int. 単位は [m]. この距離以内の時行動同期を測定する（この距離以上のとき同期していても0）． """
        beh_df2 = beh_df[[str(cow_id1), str(cow_id2)]] # 2頭を抽出
        pos_df2 = pos_df[[str(cow_id1), str(cow_id2)]] # 2頭を抽出
        # --- 行動同期スコアの計算（論文参照） --- 
        prop_vec_cow1 = self._measure_behavior_ratio(beh_df2.loc[:, str(cow_id1)].values)
        prop_vec_cow2 = self._measure_behavior_ratio(beh_df2.loc[:, str(cow_id2)].values)
        neighbor_time = self._measure_time_neighbor(pos_df2.loc[:, str(cow_id1)].values, pos_df2.loc[:, str(cow_id2)].values, threshold=10)
        dist =  np.abs(prop_vec_cow1-prop_vec_cow2).sum() # 3次元空間内での2点の距離をマンハッタン距離で求める
        score = neighbor_time * (2 - dist)
        return score

    def _measure_behavior_ratio(self, arraylist):
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

    def _measure_time_neighbor(self, arraylist1, arraylist2, threshold):
        """ 2つの位置が閾値以内にある回数を数え上げる """
        count = 0
        # 各時刻の2頭の距離を算出する
        for pos1, pos2 in zip(arraylist1, arraylist2):
            lat1, lon1 = pos1[0], pos1[1]
            lat2, lon2 = pos2[0], pos2[1]
            dist, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            if (dist <= threshold):
                count += 1
        return count / len(arraylist1) # 全時刻のうちどれくらいの割合で近くにいたかを[0,1]の範囲で表す     

    def _calculate_position_synchronization(self, pos_df, cow_id1, cow_id2, dzeta=10):
        """ 空間同期スコアを計算する 
            dzeta   : int. 単位は [m]. この距離以内の時間を計測する """
        dist_matrix = np.array([3, 6, dzeta])
        score_matrix = np.array([3, 2, 1, 0])
        df2 = pos_df[[str(cow_id1), str(cow_id2)]] # 2頭を抽出
        # --- 行動同期スコアの計算（論文参照） --- 
        score = 0
        for _, row in df2.iterrows():
            lat1, lon1, lat2, lon2 = row[0][0], row[0][1], row[1][0], row[1][1]
            dist, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            if (dist <= dist_matrix[0]):
                score += score_matrix[0]
            elif (dist <= dist_matrix[1]):
                score += score_matrix[1]
            elif (dist <= dist_matrix[2]):
                score += score_matrix[2]
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

    def _calculate_dynamic_W(self, W, leng = 5):
        """ 時系列を考慮したインタラクショングラフ（重み付きグラフ）を作成する
            W: np.array(2D) 今回のタイムスタンプでのインタラクショングラフ（重み付き無向グラフ）
            leng: int       考慮するインタラクショングラフの個数（過去分） """
        K = len(W)
        G = np.zeros((K, K)) # 今回の返却値, 重み付きグラフ
        # 今回のタイムスロットでのインタラクショングラフを追加し，古くなったインタラクショングラフを削除する
        if (len(self.community_history) < leng):
            self.community_history.append(W) # コミュニティ履歴に追加
        else:
            self.community_history.pop(0) # コミュニティ履歴からlen以上昔の要素となる0番目を削除する
            self.community_history.append(W) # コミュニティ履歴に追加
        # 時間減衰を考慮し，重み付きグラフを足し合わせることでこのタイムスロットでのクラスタリングのための重み付きグラフを作成する
        leng = len(self.community_history) if len(self.community_history) < leng else leng # コミュニティ履歴がleng未満の時
        for i in range(leng):
            zeta = (math.e ** (-1 * i)) / (i + 1) # 減衰率
            G += zeta * self.community_history[leng - i - 1]
        # 重みが負のエッジについてはゼロにリプレイス（重みの総和が0になることを防ぐため．そのまま負の重みをホールドさせた方が良いのかもしれない）
        for i in range(K):
            for j in range(K):
                if (G[i, j] < 0):
                    G[i, j] = 0
        return G


    def _visualize_graph(self, g, communities:list, date:datetime.datetime, weighted=False):
        """ インタラクショングラフを描画する """
        save_path = "./visualization/graph/" + date.strftime("%Y%m%d/")
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

    def _visualize_community(self, df, communities:list, focusing_cow_id=None):
        """ 位置情報可視化動画を作成する
            communities: list   全頭のコミュニティリスト
            focusing_cow_id: str    この牛の所属するコミュニティを先頭に配置することで色を固定する """
        # focusing_cow_idのコミュニティの並び替え
        if (focusing_cow_id is not None):
            for i, community in enumerate(communities):
                if (str(focusing_cow_id) in community):
                    community = communities.pop(i)
                    communities.insert(0, community)
                    break
        # 1頭ずつ色とキャプションを割り当てる
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

    def get_community_graph(self, communities):
        """ リスト形式のコミュニティ集合を隣接行列形式で表現し直して返す """
        N = len(self.cow_id_list)
        graph = np.zeros((N, N))
        for i, cow_id in enumerate(self.cow_id_list):
            com = []
            for community in communities:
                if (str(cow_id) in community):
                    com = community
                    break
            for j, cow_id2 in enumerate(self.cow_id_list):
                if (str(cow_id2) in com):
                    graph[i,j] = 1
                else:
                    graph[i,j] = 0
        return graph

