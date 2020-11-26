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
            try: # 最初のデータが午後12時から始まっている場合
                if (start_t == t):
                    before_t = t
                    before_lat, before_lon = float(row[1]), float(row[2])
                    before_vel = float(row[3])
                elif (start_t < t): # 最初のデータ以降
                    # 次のデータと1秒以上空きがある場合は前のデータで埋める
                    while(before_t < t):
                        revised_data.append((before_t, (before_lat, before_lon, before_vel)))
                        before_t += datetime.timedelta(seconds=1)
                    before_lat, before_lon = float(row[1]), float(row[2])
                    before_vel = float(row[3])
            except UnboundLocalError: # 午後12時からデータが始まっていない場合
                before_t = start_t
                while(before_t < t):
                    row = (before_t, (34.882, 134.86438, 0.0)) # 残りの時間は放牧場外，速度ゼロで埋める
                    revised_data.append(row)
                    before_t += datetime.timedelta(seconds=1)
                before_t = t

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
    