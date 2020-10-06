import time
import datetime
import sys, os
import numpy as np
import pandas as pd
import pdb

# my class
from synchronization.set_operation.set import SetAnalysis

class SetSeriesAnalysis:
    """ 集合（コミュニティ）の系列（時系列）を扱い、変化点を検出する機能を備えるクラス """
    cow_id_list: list # グラフのインデックスに対応する牛の個体番号を表すリスト
    set_series:list # コミュニティを表す集合のリスト

    def __init__(self, cow_id_list, set_series):
        self.cow_id_list = cow_id_list
        self.set_series = set_series

    def detect_change_point(self, target_cow_id, t_list):
        """ 変化点をコミュニティの変化から検知する（外部から呼び出されるメインのメソッド）
            target_cow_id: 変化点検知の対象とする牛の個体番号
            t_list: 時刻のリスト．インデックスに対応 """
        changepoint_list = [] # 変化点なら1, そうでなければ0
        score_list = [] # グラフ類似度のスコアを格納
        community_series = self._extract_target_community(target_cow_id) # 注目する牛のコミュニティのみを各タイムスロットごとに抽出
        # --- 変化点検知を行う ---
        for i, community in enumerate(community_series):
            if (i == 0):
                changepoint_list.append(1) # 変化点となる時刻を格納する
                previous_community = community
            else:
                set_analyzer = SetAnalysis(community, previous_community)
                is_changed = set_analyzer.compare_set()
                if (is_changed):
                    changepoint_list.append(1)
                else:
                    changepoint_list.append(0)
                previous_community = community
        return changepoint_list

    def _extract_target_community(self, target_cow_id):
        """ 注目する牛が所属するコミュニティのみを集めたリストを作る """
        ret = []
        for community_set in self.set_series:
            for community in community_set:
                if (str(target_cow_id) in community):
                    ret.append(community)
        return ret