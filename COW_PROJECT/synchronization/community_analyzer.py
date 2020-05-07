import os, sys
import pdb

class CommunityAnalyzer:
    cow_id_list:list
    day_community_list: list # コミュニティの一日のリスト
    score_dict: dict # スコアのディクショナリ、牛のIDがキーとなる

    def __init__(self, cow_id_list):
        self.cow_id_list = cow_id_list
        self.day_community_list = []
        self.score_dict = {}

    def append_community(self, community):
        self.day_community_list.append(community)

    def calculate_simpson(self):
        """ 一日のコミュニティの遷移を追ってスコアを算出する """
        for cow_id in self.cow_id_list:
            score_list = []
            com1 = []
            com2 = [] # to prevent variable from being referenced before being assigned
            for communities in self.day_community_list:
                # コミュニティの中でcow_idが所属していたコミュニティを探す
                for community in communities:
                    if (cow_id in community):
                        com2 = community
                # 一つ前のコミュニティとの差分を見る
                score = self._simpson(com1, com2)
                score_list.append(score)
                com1 = com2
            self.score_dict[str(cow_id)] = score_list

    def _simpson(self, com1:list, com2:list):
        """ シンプソン係数を計算する """
        return len(list(set(com1) & set(com2))) / max([1, min([len(com1), len(com2)])])


    def lookup_max_same_number(self):
        """ 1日で同一コミュニティに所属した回数の最大を調べる """
        self.same_num_dict = {}
        for cow_id1 in self.cow_id_list:
            count_list = []
            for cow_id2 in self.cow_id_list:
                if (cow_id1 != cow_id2):
                    count = self._number_same_member(cow_id1, cow_id2)
                    count_list.append(count)
            self.same_num_dict[cow_id1] = max(count_list)
        return

    def _number_same_member(self, cow_id1:str, cow_id2:str):
        """ 同一のコミュニティになった回数を数え上げる """
        count = 0
        for community in self.day_community_list:
            for com in community:
                if (cow_id1 in com and cow_id2 in com):
                    count += 1
        return count
    
    def get_score_dict(self):
        return self.score_dict