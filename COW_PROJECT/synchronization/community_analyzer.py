import os, sys
import datetime
import numpy as np
import pdb

class CommunityAnalyzer:
    cow_id_list:list
    communities_list:list # [time:datetime.datetime, community:list] のリスト
    graph_list:list # [time:datetime.datetime, graph:ndarray] のリスト. cow_id_listの順に行列が構成されている
    
    def __init__(self, cow_id_list):
        self.cow_id_list = cow_id_list
        self.communities_list = []
        self.graph_list = []

    def append_community(self, element):
        """ コミュニティを格納する
            element : list  [datetime, community]の要素を追加する """
        self.communities_list.append(element)
        return

    def append_graph(self, element):
        """ インタラクショングラフを格納する
            element : list  [datetime, graph:ndarray]の要素を追加する """
        self.graph_list.append(element)
        return

    def detect_change_point(self, target_list, tau=5, upsiron=1):
        """ コミュニティの変化点を検知する
            target_list: 変化点検出をしたい牛の個体番号を格納したリスト
            tau : 過去何個と現在のコミュニティを比較するか
            upsiron : 和集合をとる区画 (2分おきのコミュニティに対してupsiron=5なら10分おきのコミュニティに成形して変化点検知に用いられる)
            theta : 積集合の最小値
            eta : 和集合の最大値 """
        change_point_dict = {}
        for cow_id in target_list:
            theta, eta = 1.0, 1.0
            tmp_time_list = []
            tmp_community_list = []
            # --- cow_idの所属するコミュニティのみを取り出しリストを作る --- 
            for (t, communities) in self.communities_list:
                tmp_time_list.append(t)
                com = []
                for community in communities:
                    if (str(cow_id) in community):
                        com = community
                tmp_community_list.append(com) # 空のリスト or cow_idの所属するコミュニティのリストを格納
            time_list, community_list = self._integrate_section(tmp_time_list, tmp_community_list, upsiron)
            # --- 変化点検知を行う ---
            compared_communities = [] # 比較対象のコミュニティを格納したリスト
            change_point_list = [time_list[0]] # 変化点を格納[start_1, start_2, ..., start_n] の形のリスト
            for time, community in zip(time_list, community_list):
                change_flag = False # 変化していればTrueにする
                l = len(compared_communities)
                # パラメータの決め方は複数考えられる
                max_length, min_length = 0, 100 # theta, etaの決め方による
                for i in range(1, min(tau, l)+1):
                    com = compared_communities[l-i] # リストの後ろtau個（リストがtau以下のとき全数）と時間timeのときのcommunityを比較
                    max_length = len(com) if (max_length < len(com)) else max_length
                    min_length = len(com) if (len(com) < min_length) else min_length

                # 比較 -> フラグ
                eta = max_length * 4 / 3
                theta = min_length * 2 / 3
                for i in range(1, min(tau, l)+1):
                    com = compared_communities[l-i] # リストの後ろtau個（リストがtau以下のとき全数）とcommunityを比較
                    if (not(len(set(com) | set(community)) <= eta)): # not 演算をしているので注意
                        change_flag = True
                        break
                    if (not(len(set(com) & set(community)) >= theta)): # not 演算をしているので注意
                        change_flag = True
                        break
                
                # フラグ判定
                if (change_flag):
                    change_point_list.append(time) # --- 変化点の時間を追加 ---
                    compared_communities = [community] # 新たな比較リストを作成
                else:
                    compared_communities.append(community) # コミュニティを追加
            change_point_dict[str(cow_id)] = change_point_list
        return change_point_dict

    def _integrate_section(self, time_list, community_list, upsiron):
        """ 固定区画でコミュニティの部分和集合をとる. 返却地でそのリストを返し，フィールドは変更しない
            time_list: list             : 時間のリスト
            community_list: list(2D)    : 注目したい牛の所属するコミュニティに絞ったリスト
            upsiron: int                : 区画幅 """
        new_time_list, new_community_list = [], []
        for i, (time, community) in enumerate(zip(time_list, community_list)):
            if (i % upsiron == 0): # 区画の最初
                start = time
                union_set = set(community)
            elif (i % upsiron < upsiron - 1):
                union_set = union_set | set(community) # 和集合
            if (i % upsiron == upsiron - 1): # i % upsiron == upsiron - 1
                union_set = union_set | set(community) # 和集合
                # 新しいリストに統合した新区画を追加
                new_time_list.append(start)
                new_community_list.append(list(union_set))
        return new_time_list, new_community_list
        
    def calculate_simpson(self, target_list):
        """ 一日のコミュニティの遷移を追ってSimpsonスコアを算出する """
        score_dict = {} # スコアのディクショナリ、牛のIDがキーとなる
        for cow_id in target_list:
            score_list = []
            com1 = []
            com2 = [] # to prevent variable from being referenced before being assigned
            for (_, communities) in self.communities_list:
                # コミュニティの中でcow_idが所属していたコミュニティを探す
                for community in communities:
                    if (str(cow_id) in community):
                        com2 = community
                # 一つ前のコミュニティとの差分を見る
                score = self._simpson(com1, com2)
                score_list.append(score)
                com1 = com2
            score_dict[str(cow_id)] = score_list
        return score_dict

    def _simpson(self, com1:list, com2:list):
        """ シンプソン係数を計算する """
        return len(list(set(com1) & set(com2))) / max([1, min([len(com1), len(com2)])])

    def lookup_max_same_number(self):
        """ 1日で同一コミュニティに所属した回数の最大を調べる """
        same_num_dict = {}
        for cow_id1 in self.cow_id_list:
            count_list = []
            for cow_id2 in self.cow_id_list:
                if (cow_id1 != cow_id2):
                    count = self._number_same_member(cow_id1, cow_id2)
                    count_list.append(count)
            same_num_dict[cow_id1] = max(count_list)
        return same_num_dict

    def _number_same_member(self, cow_id1:str, cow_id2:str):
        """ 同一のコミュニティになった回数を数え上げる """
        count = 0
        for (_, communities) in self.communities_list:
            for com in communities:
                if (cow_id1 in com and cow_id2 in com):
                    count += 1
        return count

    def calculate_average_graph_density(self, start, end, target_cow_id):
        """ ある牛の所属するコミュニティのグラフ密度の平均を算出する """
        density_list = []
        for (time, graph) in self.graph_list:
            if (start <= time and time < end):
                com = self._get_particular_community(time, target_cow_id)
                # targetの所属するコミュニティのメンバのみからなるグラフに成形する
                index_list = [] # インタラクショングラフの何行目かのインデックス
                for cow_id in com:
                    index_list.append(self.cow_id_list.index(str(cow_id)))
                index_list = sorted(index_list)
                formed_graph = self._form_graph(graph, index_list)
                # グラフの密度を算出する
                density = self._calculate_graph_density(formed_graph)
                density_list.append(density)
            elif (end <= time):
                break
        return sum(density_list) / len(density_list)

    def _get_particular_community(self, time, cow_id):
        """ ある時間帯のある牛の所属するコミュニティを取得する """
        for (t, coms) in self.communities_list:
            if (t == time):
                for com in coms:
                    if (str(cow_id) in com):
                        return com

    def _form_graph(self, W, index_list):
        """ 全体のグラフ（行列）からインデックス行（列）のみを取り出して成形する """
        graph_rows = [W[index] for index in index_list]
        formed_graph = np.stack(graph_rows, axis=0)
        graph_columns = [formed_graph[:,index] for index in index_list]
        formed_graph = np.stack(graph_columns, axis=1)
        return formed_graph

    def _calculate_graph_density(self, W):
        """ グラフの密度を求める
            W:  コミュニティメンバのみに成形後のインタラクショングラフ """
        # 辺の数（重みが0より大きい）を数え上げる
        K = len(W)
        count = 0
        for i in range(K):
            for j in range(K):
                if (i < j and 0 < W[i,j]):
                    count += 1
        # 完全グラフの時の辺の数で割り密度を算出する
        density = count / (K * (K-1) / 2) if 1 < K else 0
        return density

    def get_particular_community_list(self, start, end, cow_id):
        community_list = []
        for (t, communities) in self.communities_list:
            if (start <= t and t < end):
                for com in communities:
                    if (str(cow_id) in com):
                        community_list.append(com)
            elif (end <= t):
                break
        return community_list