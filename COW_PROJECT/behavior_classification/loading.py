"""
このコードは行動分類のために行うセンサデータの読み込み処理をまとめたコードである
"""
import sys
import gc
import datetime

import cows.cow as Cow
import cows.geography as geo

"""
データベースから指定牛の指定期間のデータを読み込む (元データ (1 Hz) を5s (0.2Hz) に戻す処理も含む)
Parameter
    cow_id  : 牛の個体番号
    start, end  : 読み込む日時
Return
    time_list   : 時間の2次元リスト（1日分のデータ × 指定日数分）
    position_list   : 緯度・経度の3次元リスト（(緯度, 経度) × 1日分のデータ × 指定日数分）
    distance_list   : 距離の2次元リスト（1日分のデータ × 指定日数分）
    velocity_list   : 速さの2次元リスト（1日分のデータ × 指定日数分）
    angle_list      : 角度の2次元リスト（1日分のデータ × 指定日数分）
    ※　ここでは緯度と経度だけを読み込み前処理で距離や速さを計算する方が良いかもしれません
"""
def load_gps(cow_id, start, end):
    print(sys._getframe().f_code.co_name, "実行中")
    print("GPSの読み込みを開始します---")
	# データを読み込み，それぞれのリストを作成 (情報ごとのリストにするか時間ごとのリストのリストにするかは場合による) 
    time_list = [] # 時間のリスト (主キー)
    position_list = [] # (緯度，経度) のリスト
    distance_list = [] # 距離のリスト
    velocity_list = [] # 速さのリスト
    angle_list = [] # 移動角度のリスト
    dt = datetime.datetime(start.year, start.month, start.day)
    a = start
    while(dt < end):
        t_list = []
        pos_list = []
        dis_list = []
        vel_list = []
        ang_list = []
        cow = Cow.Cow(cow_id, dt)
        dt = dt + datetime.timedelta(days = 1)
        while(a <= dt and a < end):
            gps_list = cow.get_gps_list(a, a + datetime.timedelta(minutes = 60))
            g_before = None
            for i in range(int(len(gps_list) / 5)):
                g = gps_list[i * 5]
                if g_before is not None:
                    lat1, lon1, vel1 = g_before.get_gps_info(g_before.get_datetime())
                    lat2, lon2, vel2 = g.get_gps_info(g.get_datetime())
                    distance, angle = geo.get_distance_and_direction(lat1, lon1, lat2, lon2, False)
                    #print(g.get_datetime().strftime("%Y/%m/%d %H:%M:%S") + " : ", lat2 , ",", lon2)
                    t_list.append(g.get_datetime()) #時間の格納
                    pos_list.append(geo.translate(lat2, lon2)) #位置情報の格納
                    dis_list.append(distance) #距離の格納
                    vel_list.append(vel2) #速さの格納
                    ang_list.append(angle) #角度の格納
                g_before = g
            a = a + datetime.timedelta(minutes = 60)
            del gps_list
            gc.collect()
        time_list.append(t_list) #1日分の時間のリストの格納
        position_list.append(pos_list) #1日分の位置情報の格納
        distance_list.append(dis_list) #1日分の距離のリストの格納
        velocity_list.append(vel_list) #1日分の速さのリストの格納
        angle_list.append(ang_list) #1日分の角度のリストの格納
        del cow
        gc.collect()
        a = dt
    print("---GPSの読み込みが終了しました")
    print(sys._getframe().f_code.co_name, "正常終了")
    print("The length of time_list: ", len(time_list), "day(s)\n")
    return time_list, position_list, distance_list, velocity_list, angle_list

"""
データの解析に使用する時間 (午後12時-午前9時 JST) 分を抽出する
Parameter
    time_list   : 時間の2次元リスト（1日分のデータ × 指定日数分）
    position_list   : 緯度・経度の3次元リスト（(緯度, 経度) × 1日分のデータ × 指定日数分）
    distance_list   : 距離の2次元リスト（1日分のデータ × 指定日数分）
    velocity_list   : 速さの2次元リスト（1日分のデータ × 指定日数分）
    angle_list      : 角度の2次元リスト（1日分のデータ × 指定日数分）
Return
    new_time_list   : 抽出後の時間の2次元リスト（1日分のデータ × 指定日数分）
    new_position_list   : 抽出後の緯度・経度の3次元リスト（(緯度, 経度) × 1日分のデータ × 指定日数分）
    new_distance_list   : 抽出後の距離の2次元リスト（1日分のデータ × 指定日数分）
    new_velocity_list   : 抽出後の速さの2次元リスト（1日分のデータ × 指定日数分）
    new_angle_list      : 抽出後の角度の2次元リスト（1日分のデータ × 指定日数分）
"""
def select_used_time(time_list, position_list, distance_list, velocity_list, angle_list):
    print(sys._getframe().f_code.co_name, "実行中")
    print("12:00 pm - 9:00 amのデータの抽出を開始します---")
    knot = 0.514444 # 1 knot = 0.51444 m/s
    new_time_list = []
    new_position_list = []
    new_distance_list = []
    new_velocity_list = []
    new_angle_list = []
    for (t, p, d, v, a) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
        t = t + datetime.timedelta(hours = 9)
        if(t.hour < 9 or 12 < t.hour):
            new_time_list.append(t)
            new_position_list.append(p)
            new_distance_list.append(d) 
            new_velocity_list.append(v * knot) #単位を[m/s]に直しているだけ
            new_angle_list.append(a)
    print("---12:00 pm - 9:00 amのデータの抽出が終了しました")
    print(sys._getframe().f_code.co_name, "正常終了")
    print("The length of time_list: ", len(new_time_list), "\n")
    return new_time_list, new_position_list, new_distance_list, new_velocity_list, new_angle_list

"""
特徴抽出したCSVファイル (圧縮済み) から元の時系列データを作成する
Parameter
    filename    : csvファイルのパス (str)
"""