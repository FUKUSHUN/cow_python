import datetime
import os,sys
import pandas as pd
import numpy as np
import pdb

# 自作クラス
import position_information.position_loader as position_loader

class Synchronizer:
    date: datetime.datetime
    cow_id_list: list
    cows_data: pd.DataFrame
    
    def __init__(self, date:datetime.datetime, cow_id_list:list):
        self.date = date
        self.cow_id_list = self._confirm_csv(cow_id_list)
        self._make_position_df()

    def _confirm_csv(self, cow_id_list):
        """ 行動分類のファイルが存在しているか確認し，なければIDのリストから削除する """
        dir_path = "./position_information/" + self.date.strftime("%Y%m%d/")
        delete_list = []
        for cow_id in cow_id_list:
            filepath = dir_path + str(cow_id) + ".csv"
            if (not os.path.isfile(filepath)):
                cow_id_list.remove(cow_id)
                delete_list.append(cow_id)
        print("位置情報ファイルの存在しない牛のIDをリストから削除しました. 削除した牛のリスト: ", delete_list)
        return cow_id_list
    
    def _make_position_df(self):
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
            loader = position_loader.PositionLoader(cow_id, self.date)
            data = loader.get_data()
            df = pd.concat([df, pd.Series(data=data.values[:,1], name=cow_id)], axis=1)
            print("データを結合しました", cow_id)
        df = df.set_index('Time') # 時間をインデックスにする
        self.cows_data = df

    def extract_df(self, start:datetime.datetime, end:datetime.datetime, delta: int):
        """ 特定の時間のデータを抽出する """
        df2 = self.cows_data[(start <= self.cows_data.index) & (self.cows_data.index < end)] # 抽出するときは代わりの変数を用意すること
        return df2[::delta] # 等間隔で抽出する