import os, sys
import re
import datetime
import numpy as np
import pandas as pd
import pdb

class Loader:
    gps_id: str

    def __init__(self, gps_id):
        self.gps_id = str(gps_id)

    def load(self, date:datetime.datetime):
        # GPSデータがあるディレクトリ
        gps_dir = os.path.expanduser("~") + "\\Dropbox\\GPSデータ\\IDA-GPS\\"
        filename = gps_dir + self.gps_id + "\\" + date.strftime("%y%m%d") + ".txt"
        try:
            with open(filename) as f:
                lines = f.readlines()
                data_list = []
                for line in lines:
                    if (bool(re.search("\$GPRMC", line))):
                        data_list.append(line)
            return data_list
        except FileNotFoundError:
            return None


if __name__ == "__main__":
    # start and end
    start = datetime.datetime(2020, 7, 1, 0, 0, 0)
    end = datetime.datetime(2020, 7, 8, 0, 0, 0)
    # バッテリー記録を見る
    record_filename = "./GPSーバッテリーチェック.xlsx"
    df = pd.read_excel(record_filename, sheet_name=start.strftime("%m%d"), index_col=[1,2], header=0)
    df = df.dropna(subset=['V1', 'V2'])

    # 読み込み
    for indexes, _ in df.iterrows():
        loader = Loader(str(indexes[0]))
        date = start
        while (date <= end):
            lines = loader.load(date)
            if (lines is not None):
                df.loc[indexes,date] = len(lines)
            else:
                df.loc[indexes,date] = None
            date += datetime.timedelta(days=1)
            # pdb.set_trace()
    df.to_csv(start.strftime("%m%d.csv"))