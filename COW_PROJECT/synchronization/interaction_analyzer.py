import datetime
import numpy as np
import pandas as pd

import pdb

# 自作メソッド
import cows.geography as geography
# 自作クラス
import behavior_information.synchronizer as behavior_synchronizer # 行動同期
import position_information.synchronizer as position_synchronizer # 空間同期
import visualization.place_plot as place_plot

class InteractionAnalyzer:
    """ 1頭の牛に着目したときのインタラクション分析を行う """
    cow_id: int # 対象とする牛
    behavior_synch: behavior_synchronizer.Synchronizer
    position_synch: position_synchronizer.Synchronizer

    def __init__(self, cow_id:int, behavior_synch: behavior_synchronizer.Synchronizer, position_synch: position_synchronizer.Synchronizer):
        """ 分析の対象とする牛の個体番号を設定 """
        self.cow_id = cow_id
        self.behavior_synch = behavior_synch
        self.position_synch = position_synch

    def extract_feature(self, start:datetime.datetime, end:datetime.datetime, community):
        """ 特徴を抽出し，その時間幅での各特徴をリストとして出力
            Parameter
                df: pd.DataFrame    時間幅で抽出された行動と位置のすべての牛のデータ (1秒ごとのデータではないので注意)
                community: list     対象とする牛が含まれるコミュニティ
            Return
                features: 特徴のリスト """
        # 自分の牛のデータのみを抽出
        both_df, beh_df, pos_df = self._extract_and_merge_df(start, end, delta = 5)
        # 時間特徴
        time_interval = (end - start).total_seconds() / 60 # 単位を分にする
        # 行動特徴
        my_beh_df = beh_df[[str(self.cow_id)]]
        behavior_ratio = self._measure_behavior_ratio(my_beh_df)
        # 距離特徴
        my_pos_df = pos_df[[str(self.cow_id)]]
        total_distance = self._measure_mileage(my_pos_df)
        # コミュニティサイズ
        community_size = len(community)
        # 最も距離の近かった牛（平均）との距離
        minimum_dist_cow, minimum_dist = self._find_minimun_distance(pos_df, community)
        features = [time_interval, total_distance, behavior_ratio['rest'], behavior_ratio['graze'], behavior_ratio['walk'], community_size, minimum_dist_cow, minimum_dist]
        self._visualize_adjectory(pos_df, [str(self.cow_id), str(minimum_dist_cow)])
        return features

    def _extract_and_merge_df(self, start, end, delta=5):
        """ startからendまでの時間のデータをdeltaごとにスライスして抽出し，行動，空間の2つのデータを結合する(どちらも1秒ごとに成形し，インデックスがTimeになっている前提)
            delta   : int. 単位は[s (個)]. この個数ごとに等間隔でデータをスライス """
        beh_df = self.behavior_synch.extract_df(start, end, delta)
        pos_df = self.position_synch.extract_df(start, end, delta)
        merged_df = pd.concat([beh_df, pos_df], axis=1)
        return merged_df, beh_df, pos_df

    def _measure_behavior_ratio(self, df):
        """ 行動割合を算出する """
        behavior_0, behavior_1, behavior_2 = 0, 0, 0 # カウントアップ変数
        # 各時刻の行動を順番に走査してカウント
        for _, row in df.iterrows():
            if (row[0][1] == 0):
                behavior_0 += 1
            elif (row[0][1] == 1):
                behavior_1 += 1
            else:
                behavior_2 += 1
        proportion = {}
        proportion["rest"] = behavior_0 / (behavior_0 + behavior_1 + behavior_2)
        proportion["graze"] = behavior_1 / (behavior_0 + behavior_1 + behavior_2)
        proportion["walk"] = behavior_2 / (behavior_0 + behavior_1 + behavior_2)
        return proportion

    def _measure_mileage(self, df):
        """ 総移動距離（各時刻の移動距離の総和）を算出する """
        mileage = 0
        before_lat, before_lon = None, None
        for _, row in df.iterrows():
            lat, lon = row[0][0], row[0][1]
            if ((before_lat is not None) and (before_lon is not None)):
                dis, _ = geography.get_distance_and_direction(before_lat, before_lon, lat, lon, True)
                mileage += dis
            before_lat, before_lon = lat, lon
        return mileage

    def _find_minimun_distance(self, df, community):
        """ 最も総距離の短い牛との間の距離の平均を求める """
        minimum_dist_cow, minimum_dist = None, 100000
        if (len(community) == 1):
            return "None", 0 # コミュニティメンバがいない場合は欠損値扱い
        else:
            my_df = df[[str(self.cow_id)]]
            for cow_id in community:
                if (cow_id != str(self.cow_id)):
                    opponent_df = df[[str(cow_id)]]
                    merged_df = pd.concat([my_df, opponent_df], axis=1)
                    sum_dis = 0
                    for _, row in merged_df.iterrows():
                        lat1, lon1, lat2, lon2 = row[0][0], row[0][1], row[1][0], row[1][1]
                        dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
                        sum_dis += dis
                    if(sum_dis < minimum_dist):
                        minimum_dist = sum_dis
                        minimum_dist_cow = cow_id
            return minimum_dist_cow, minimum_dist / len(df) # 1データ当たりの距離の平均を算出

    def _visualize_adjectory(self, df, community):
        """ 軌跡描画を行う
            df: pd.DataFrame    pos_df  
            community: list self.cow_idを含む牛のコミュニティメンバのリスト """
        caption_list = []
        color_list = []
        focusing_cow_id = str(self.cow_id)
        if ("None" in community):
            community.remove("None")
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
