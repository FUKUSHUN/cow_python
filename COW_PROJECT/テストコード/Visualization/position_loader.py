import datetime
import os,sys
import pandas as pd
import numpy as np
import pdb

class PositionLoader:
    cow_id: int
    data: pd.DataFrame # (Time, Latitude, Longitude, Velocity)

    def __init__(self, cow_id:int, date:datetime.datetime):
        self.cow_id = cow_id
        self._load_data(date)

    def _load_data(self, date:datetime.datetime):
        filename = "./position_information/" + date.strftime("%Y%m%d/") + str(self.cow_id) + ".csv"
        df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,1,2,3,4], names=['index', 'Time', 'Latitude', 'Longitude', 'Velocity'],index_col=0) # csv読み込み
        # --- 1秒ごとのデータになっているデータの欠損値を補間してself.dataに登録 ---
        revised_data = [] # 新規登録用リスト
        start_t = datetime.datetime.strptime(date.strftime("%Y%m%d"), "%Y%m%d") + datetime.timedelta(hours=12) # 正午12時を始まりとする
        for _, row in df.iterrows():
            t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") # datetime
            if (start_t == t):
                before_t = t
                before_lat, before_lon = float(row[1]), float(row[2])
                before_vel = float(row[3])
            elif (start_t < t):
                # 次のデータと1秒以上空きがある場合は前のデータで埋める
                while(before_t < t):
                    revised_data.append((before_t, (before_lat, before_lon, before_vel)))
                    before_t += datetime.timedelta(seconds=1)
                before_lat, before_lon = float(row[1]), float(row[2])
                before_vel = float(row[3])
        # 最後のデータを格納し、午前9時までに空きのデータがある場合は適当な数値で埋める
        end_t = datetime.datetime.strptime(date.strftime("%Y%m%d"), "%Y%m%d") + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌日午前9時を終わりとする
        while (before_t < end_t):
            t = before_t
            if (before_t == t):
                row = (before_t, (before_lat, before_lon, before_vel)) # 最後のtをまだ登録していないので登録する
            else:
                row = (before_t, (34.882, 134.86438, 0.0)) # 残りの時間は放牧場外，速度ゼロで埋める
            revised_data.append(row)
            before_t += datetime.timedelta(seconds=1)
        self.data = pd.DataFrame(revised_data, columns=['Time', 'Data'])
        return

    def get_data(self):
        return self.data
    

class Synchronizer:
    date: datetime.datetime
    cow_id_list: list
    cows_data: pd.DataFrame
    
    def __init__(self, date:datetime.datetime, cow_id_list:list):
        self.date = date
        self.cow_id_list = self._confirm_csv(cow_id_list)
        self._make_position_df()
        pdb.set_trace()

    def _confirm_csv(self, cow_id_list):
        """ 行動分類のファイルが存在しているか確認しなければIDのリストから削除する """
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
            loader = PositionLoader(cow_id, self.date)
            data = loader.get_data()
            df = pd.concat([df, pd.Series(data=data.values[:,1], name=cow_id)], axis=1)
            print("データを結合しました", cow_id)
        self.cows_data = df

    def _extract_df(self, df:pd.DataFrame, start:datetime.datetime, end:datetime.datetime, delta: int):
        """ 特定の時間のデータを抽出する """
        df2 = df[(start <= df["Time"]) & (df["Time"] < end)] # 抽出するときは代わりの変数を用意すること
        return df2[::delta] # 等間隔で抽出する