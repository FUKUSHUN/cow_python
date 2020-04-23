import os, sys
import pdb

class CommunityAnalyzer:
    cow_id_list:list
    day_community_list: list # コミュニティの一日のリスト
    same_num_dict: dict # 同じコミュニティになった回数を格納する（最大値か平均値か）

    def __init__(self, cow_id_list):
        self.cow_id_list = cow_id_list
        self.day_community_list = []

    def append_community(self, community):
        self.day_community_list.append(community)

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
    
    def get_same_num_dict(self):
        return self.same_num_dict