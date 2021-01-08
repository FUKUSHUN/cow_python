"""
このコードは行動分類のために行うセンサデータの読み込み処理をまとめたコードである
"""
import sys
import gc
import datetime
import pdb

import cows.cow as Cow
import cows.geography as geo

def load_gps(cow_id, date):
    """ データベースから指定牛の指定期間のデータを読み込む (元データ (1 Hz) を5s (0.2Hz) に戻す処理も含む)
        Parameter
            cow_id  : 牛の個体番号
            date  : datetime, 読み込む日時
        Return
            time_list   : 時間の2次元リスト（1日分のデータ）
            position_list   : 緯度・経度の3次元リスト（(緯度, 経度) × 1日分のデータ）
            distance_list   : 距離の2次元リスト（1日分のデータ）
            velocity_list   : 速さの2次元リスト（1日分のデータ）
            angle_list      : 角度の2次元リスト（1日分のデータ）
            ※　ここでは緯度と経度だけを読み込み前処理で距離や速さを計算する方が良いかもしれません """
    print(sys._getframe().f_code.co_name, "実行中")
    print("GPSの読み込みを開始します---")
	# データを読み込み，それぞれのリストを作成 (情報ごとのリストにするか時間ごとのリストのリストにするかは場合による) 
    time_list = [] # 時間のリスト (主キー)
    position_list = [] # (緯度，経度) のリスト
    distance_list = [] # 距離のリスト
    velocity_list = [] # 速さのリスト
    angle_list = [] # 移動角度のリスト
    dt = datetime.datetime(date.year, date.month, date.day)
    cow = Cow.Cow(cow_id, dt)
    dt = dt + datetime.timedelta(hours=9) # JSTでデータベース登録されているため時間を合わせる
    gps_list = cow.get_gps_list(dt, dt + datetime.timedelta(hours=24)) # 1日分のGPSデータを読み込む
    g_before = None
    for i in range(int(len(gps_list) / 5)):
        g = gps_list[i * 5] # 5の倍数の要素の分だけデータを取り出す
        if (g_before is not None):
            lat1, lon1, vel1 = g_before.get_gps_info(g_before.get_datetime())
            lat2, lon2, vel2 = g.get_gps_info(g.get_datetime())
            distance, angle = geo.get_distance_and_direction(lat1, lon1, lat2, lon2, False)
            time_list.append(g.get_datetime()) #時間の格納
            position_list.append(geo.translate(lat2, lon2)) #位置情報の格納
            distance_list.append(distance) #距離の格納
            velocity_list.append(vel2) #速さの格納
            angle_list.append(angle) #角度の格納
        g_before = g
    print("---GPSの読み込みが終了しました")
    print(sys._getframe().f_code.co_name, "正常終了")
    print("The length of time_list: ", len(time_list), "(data)\n")
    return time_list, position_list, distance_list, velocity_list, angle_list

def select_used_time(time_list, position_list, distance_list, velocity_list, angle_list, date):
    """ データの解析に使用する時間 (午後12時-午前9時 JST) 分を抽出する, 厳密には5分分余分に取り出す（行動分類時に切りおとされるため）
        Parameter
            time_list   : 時間の2次元リスト（1日分のデータ）
            position_list   : 緯度・経度の3次元リスト（(緯度, 経度) × 1日分のデータ）
            distance_list   : 距離の2次元リスト（1日分のデータ）
            velocity_list   : 速さの2次元リスト（1日分のデータ）
            angle_list      : 角度の2次元リスト（1日分のデータ）
        Return
            new_time_list   : 抽出後の時間の2次元リスト（1日分のデータ）
            new_position_list   : 抽出後の緯度・経度の3次元リスト（(緯度, 経度) × 1日分のデータ）
            new_distance_list   : 抽出後の距離の2次元リスト（1日分のデータ）
            new_velocity_list   : 抽出後の速さの2次元リスト（1日分のデータ）
            new_angle_list      : 抽出後の角度の2次元リスト（1日分のデータ）"""
    print(sys._getframe().f_code.co_name, "実行中")
    print("12:00 pm - 9:00 amのデータの抽出を開始します---")
    #knot = 0.514444 # 1 knot = 0.51444 m/s
    new_time_list = []
    new_position_list = []
    new_distance_list = []
    new_velocity_list = []
    new_angle_list = []
    start = datetime.datetime(date.year, date.month, date.day, 11, 55, 0) # その日の11:55
    end = start + datetime.timedelta(hours=21) + datetime.timedelta(minutes=5) # 翌朝9:00
    for (t, p, d, v, a) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
        if(start < t and t < end):
            new_time_list.append(t)
            new_position_list.append(p)
            new_distance_list.append(d) 
            new_velocity_list.append(v) #単位はすでに[m/s]になっている
            new_angle_list.append(a)
    print("---12:00 pm - 9:00 amのデータの抽出が終了しました")
    print(sys._getframe().f_code.co_name, "正常終了")
    print("The length of time_list: ", len(new_time_list), "\n")
    return new_time_list, new_position_list, new_distance_list, new_velocity_list, new_angle_list


def make_time_list(date):
    """ t_listを作成する """
    time_list = []
    time = date
    end = date + datetime.timedelta(days=1)
    while(time < end):
        t = time + datetime.timedelta(hours = 9)
        if(t.hour < 9 or 11 < t.hour):
            time_list.append(t)
        time += datetime.timedelta(seconds=5)
    print("---12:00 pm - 9:00 amのデータの抽出が終了しました")
    return time_list