import os, sys
import pdb

class CommunityAnalyzer:
    cow_id_list:list
    communities_list:list # [time:datetime.datetime, community:list] のリスト
    
    def __init__(self, cow_id_list):
        self.cow_id_list = cow_id_list
        self.communities_list = []

    def append_community(self, element):
        """ コミュニティを格納する
            element : list  [datetime, community]の要素を追加する """
        self.communities_list.append(element)
        return

    def detect_change_point(self, target_list, tau=6):
        """ コミュニティの変化点を検知する
            tau : 過去何個と現在のコミュニティを比較するか
            theta : 積集合の最小値
            eta : 和集合の最大値 """
        change_point_dict = {}
        for cow_id in target_list:
            theta, eta = 1.0, 1.0
            time_list = []
            community_list = []
            # --- cow_idの所属するコミュニティのみを取り出しリストを作る --- 
            for (t, communities) in self.communities_list:
                time_list.append(t)
                com = []
                for community in communities:
                    if (str(cow_id) in community):
                        com = community
                community_list.append(com) # 空のリスト or cow_idの所属するコミュニティのリストを格納
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
                eta = max_length * 6 / 5
                theta = min_length * 4 / 5
                for i in range(1, min(tau, l)+1):
                    com = compared_communities[l-i] # リストの後ろtau個（リストがtau以下のとき全数）とcommunityを比較
                    if (not(len(set(com) | set(community)) <= eta)):
                        change_flag = True
                        break
                    if (not(len(set(com) & set(community)) >= theta)):
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
    