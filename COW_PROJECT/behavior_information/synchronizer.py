import os, sys
import datetime
import numpy as np
import pandas as pd
import networkx as nx # Louvain法
import community # Louvain法
import pdb

# 自作クラス
import behavior_information.behavior_loader as behavior_loader

class Synchronizer:
    date: datetime.datetime
    cow_id_list: list
    cows_data: pd.DataFrame # (vel, behavior_label) 12:00 - 9:00まで1秒ずつ
    score_dict: dict # 行動同期スコアを格納, cow_id_combination_listのインデックスをキーにする

    def __init__(self, date, cow_id_list):
        self.date = date
        self.cow_id_list = self._confirm_csv(cow_id_list)
        self._make_behavior_df()

    def _confirm_csv(self, cow_id_list):
        """ 行動分類のファイルが存在しているか確認しなければIDのリストから削除する """
        dir_path = "./behavior_information/" + self.date.strftime("%Y%m%d/")
        delete_list = []
        for cow_id in cow_id_list:
            filepath = dir_path + str(cow_id) + ".csv"
            if (not os.path.isfile(filepath)):
                delete_list.append(cow_id)
        # イテレーションの最中に要素を削除すると反復が崩れるのであとから一括で削除する
        for cow_id in delete_list:
            cow_id_list.remove(cow_id)
        print("行動分類ファイルの存在しない牛のIDをリストから削除しました. 削除した牛のリスト: ", delete_list)
        return cow_id_list

    def _make_behavior_df(self):
        """ 時刻をキーにして各列に牛の列をもつdataframeを生成する """
        # 時刻は統一されているので先にキーを作成する
        print("牛のデータを結合します", self.date)
        time_list = []
        start = self.date + datetime.timedelta(hours=12) # 正午12時から始める
        end = self.date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌9時を終わりに設定している
        dt = start
        while (dt < end):
            time_list.append(dt)
            dt += datetime.timedelta(seconds=1)
        # 牛のID順にデータを追加していく
        df = pd.Series(data=time_list, name='Time')
        for cow_id in self.cow_id_list:
            loader = behavior_loader.BehaviorLoader(cow_id, self.date)
            data = loader.get_data()
            df = pd.concat([df, pd.Series(data=data.values[:,1], name=cow_id)], axis=1)
            print("データを結合しました", cow_id)
        df = df.set_index('Time') # 時間をインデックスにする
        self.cows_data = df
        return

    def extract_df(self, start:datetime.datetime, end:datetime.datetime, delta: int):
        """ 特定の時間のデータを抽出する """
        df2 = self.cows_data[(start <= self.cows_data.index) & (self.cows_data.index < end)] # 抽出するときは代わりの変数を用意すること
        return df2[::delta] # 等間隔で抽出する
    
    def get_cow_id_list(self):
        return self.cow_id_list