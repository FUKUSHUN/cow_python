# -*- encoding: utf-8 -*-
import datetime
import random
import sys
import os
import csv

# 自作クラス
import myClass.feature_extraction as feature_extraction

""" このプログラムでは教師データとなるデータを複数生成する
    セグメント化を行い，特徴作成を行い，ファイルに出力する
    選ばれる日付及び牛はランダムに選択する
    それを複数回おこなう 
    ※注意
    本プログラムファイルの階層が他のメインファイルと異なるためgps_nmea_data_list.pyの_db_file_pathを書き換える必要あり """

# 別ファイルでモジュール化
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

if  __name__ == '__main__':
    os.chdir('../') # 諸々の諸事情によりカレントディレクトリを1階層上に移動
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/"
    filepath = "./behavior_classification/training_data/random_extraction/"
    initial_date = datetime.datetime(2018, 12, 20, 0, 0, 0)
    for i in range(50):
        rnd = random.randint(0, 193) # 0 - xx の間のランダムな数字を生成
        date = initial_date + datetime.timedelta(days=rnd) # ランダムな数だけ日にちをずらす
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        rnd2 = random.randint(0, len(cow_id_list)-1) # a <= n <= bなので注意
        cow_id = cow_id_list[rnd2]
        savefile = filepath + date.strftime("%Y%m%d") + "_" + str(cow_id) + ".csv"
        print(date.strftime("%Y/%m/%d") + "の牛" + str(cow_id) + "のデータを" + savefile + "に出力します")
        output_features = feature_extraction.FeatureExtraction(savefile, date, cow_id)
        output_features.output_features()