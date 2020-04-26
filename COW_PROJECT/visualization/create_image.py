import os,sys
import datetime
import csv
import numpy as np
import pandas as pd
import pdb

# 自作クラス
os.chdir('../') # カレントディレクトリを./COW_PROJECT/へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import visualization.place_plot as place_plot
import position_information.synchronizer as position_synchronizer

def get_existing_cow_list(date:datetime, filepath):
    """ 引数の日にちに第一放牧場にいた牛のリストを得る """
    filepath = filepath + date.strftime("%Y-%m") + ".csv"
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if (datetime.datetime.strptime(row[0], "%Y/%m/%d") == date):
                return row[1:]
    print("指定の日付の牛のリストが見つかりません", date.strftime("%Y/%m/%d"))
    sys.exit()

if __name__ == '__main__':
    delta = 5 # 見る間隔 [sec]
    start = datetime.datetime(2018, 10, 1, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    end = datetime.datetime(2018, 10, 2, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル

    date = start
    while (date < end):
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        synch = position_synchronizer.Synchronizer(date, cow_id_list) # 時間と牛のIDを元にした位置情報のマトリックスを作る
        dt = date + datetime.timedelta(hours=12) # 正午12時を始まりにする
        end_dt = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌9時を終わりにする
        while (dt < end_dt):
            df = synch.extract_df(dt, dt+datetime.timedelta(minutes=5), delta)
            maker = place_plot.PlotMaker()
            maker.make_movie(df)
            dt += datetime.timedelta(minutes=5)
            pdb.set_trace()
        date += datetime.timedelta(days=1) # 時間を進める