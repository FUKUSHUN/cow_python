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
import cows.cowshed as cowshed


def confirm_dir(dir_path):
    """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
    if (os.path.isdir(dir_path)):
        return
    else:
        os.makedirs(dir_path)
        print("ディレクトリを作成しました", dir_path)
        return

def output_csv(filepath, pos_list:list):
    """ 位置情報のリストをCSVファイルとして出力する """
    time_list, lat_list, lon_list, vel_list = [], [], [], []
    for pos in pos_list:
        dt = pos.get_datetime()
        lat, lon, vel = pos.get_gps_info(dt)
        lat, lon = translate(lat, lon)
        time_list.append(dt)
        lat_list.append(lat)
        lon_list.append(lon)
        vel_list.append(vel)
    if (len(time_list) != 0):
        output_df = pd.concat([pd.Series(data=time_list, name='Time'), pd.Series(data=lat_list, name='Latitude'), pd.Series(data=lon_list, name='Longitude'), pd.Series(data=vel_list, name='Velocity')], axis=1)
        output_df.to_csv(filepath)
        print("位置情報のCSVファイルを出力しました", filepath)
    else:
        print("位置情報が存在しません。", filepath)
    return

#経度・緯度をdddmm.mmmm表記をddd.dddd形式に変換する (他のファイルから呼び出されることもあり)
def translate(lat, lon):
	lat_d = lat // 100
	lon_d = lon // 100
	lat_m = (lat % 100) / 60
	lon_m = (lon % 100) / 60
	return lat_d + lat_m, lon_d + lon_m

if __name__ == '__main__':
    start = datetime.datetime(2018, 10, 1, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    end = datetime.datetime(2018, 11, 1, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    output_dirpath = "./position_information/" # 分析用のファイル

    date = start
    while (date < end):
        cows = cowshed.Cowshed(date) # その日の牛の集合
        cow_df = cows.get_cow_list(date+datetime.timedelta(hours=12), date + datetime.timedelta(days=1)+datetime.timedelta(hours=9))
        cow_id_list = np.ravel(cow_df[0:1].values.tolist()) # データのある牛のIDを取得
        # 時間と牛のIDを元にした位置情報のマトリックスを作る (欠損に対応するため，短い間隔でつくっていく)
        for _, item in cow_df.iteritems():
            output_filepath = output_dirpath + date.strftime("%Y%m%d/")
            confirm_dir(output_filepath)
            output_filepath += str(item[0]) + ".csv"
            output_csv(output_filepath, item[1])
        date += datetime.timedelta(days=1)