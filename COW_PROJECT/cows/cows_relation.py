#-*- encoding:utf-8 -*-
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cow #自作クラス
import cowshed #自作クラス
import geography #自作クラス
import gps.gps_nmea_data_list as gpslist#自作クラス
import gps.gps_nmea_data as gpsdata#自作クラス

class CowsRelation:
    main_cow_id:int #関係を見る対象の牛の個体番号
    main_cow_data:gpslist.GpsNmeaDataList ##関係を見る対象の牛のGPSデータ
    cow_data:pd.DataFrame #牛のリスト (現時点ではpandasのデータフレーム型でidとGPS履歴を保持, main_cowも含む) 

    """
        dfの形式はcowshed.Cowshed.get_cow_list()を参照すること
        ------------------------
        ID      | 20... | 20...
        ------------------------
        Data    | [...] | [...]
        ------------------------
        の形をしている (1/29時点)
        行と列を逆にする方が良いかもしれない. 速さとアクセスのしやすさのトレードオフ？
    """
    def __init__(self, id, data, df):
        self.main_cow_id = id
        self.main_cow_data = data
        self.cow_data = df

    def calc_weight(self):
        for i in range(len(self.cow_data.columns)):
            if(self.cow_data.iloc[0, i] == self.main_cow_id):
                self.make_distance_data(self.main_cow_data, self.cow_data.iloc[1, i])
        return

    def calc_evaluation_value(self):
        for i in range(len(self.cow_data.columns)):
            if(self.cow_data.iloc[0, i] == self.main_cow_id):
                self.make_cosine_data(self.main_cow_data, self.cow_data.iloc[1, i])
        return

    """
        gpsの各時刻での距離を求める
        g_list1 :   gpslist
        g_list2 :   gpslist
    """
    def make_distance_data(self, g_list1:gpslist.GpsNmeaDataList, g_list2:gpslist.GpsNmeaDataList):
        distance_list = []
        i1 = 0 #g_list2のインデックス
        i2 = 0 #g_list2のインデックス
        while(i1 < len(g_list1) and i2 < len(g_list2)):
            t = g_list1[i1].get_datetime()
            if(t == g_list2[i2].get_datetime()):
                lat1, lon1, _ = g_list1[i1].get_gps_info(t)
                lat2, lon2, _ = g_list2[i2].get_gps_info(g_list2.get_datetime())
                d, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2)
                i1 += 1
                i2 += 1
                distance_list.append((t, d)) #左に時刻，右に距離
            elif(t > g_list2[i2].get_datetime()):
                i2 = i2 + 1
            else:
                i1 = i1 + 1
        return distance_list

    """
        gpsの各時刻での被接近度を求める
        g_list1 :   gpslist (main)
        g_list2 :   gpslist (other)
        listのインデックス順にたどっているのでiとi + 1の関係が時間軸では必ずしも直近とは限らないので注意
        dfがごく短い時間 (例えば10分) 間隔でつくられることを想定して，データが欠けまくっていることはないだろう (あっても1回くらい) と考えてこのようにしている
    """
    def make_cosine_data(self, g_list1:gpslist.GpsNmeaDataList, g_list2:gpslist.GpsNmeaDataList):
        cosine_list = []
        i1 = 0 #g_list2のインデックス
        i2 = 0 #g_list2のインデックス
        while(i1 < len(g_list1) and i2 < len(g_list2)):
            t = g_list1[i1].get_datetime()
            if(t == g_list2[i2].get_datetime()):
                if(i1 != len(g_list1) - 1 and i2 != len(g_list2) - 1):
                    cos = geography.get_cos_sim(g_list1[i1 + 1], g_list2[i2 + 1], g_list1[i1], g_list2[i2])
                    i1 += 1
                    i2 += 1
                    cosine_list.append((t, cos)) #左に時刻，右にコサイン類似度
                else:
                    i1 += 1
                    i2 += 1
            elif(t > g_list2[i2].get_datetime()):
                i2 = i2 + 1
            else:
                i1 = i1 + 1
        return cosine_list

