#-*- encoding:utf-8 -*-
import pandas as pd
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cow #自作クラス
import cowshed #自作クラス
import geography #自作クラス
import gps.gps_nmea_data_list as gpslist#自作クラス
import gps.gps_nmea_data as gpsdata#自作クラス

class TwoCowsRelation:
    cow_gps_list1:gpslist.GpsNmeaDataList
    cow_gps_list2:gpslist.GpsNmeaDataList
    """
        2頭間の関係を見るクラスで二つのgps_listが必要
        CowsRelationの分社化みたいなもの (コミュニティ生成などで外部から呼び出されることもある)
    """
    def __init__(self, g_list1, g_list2):
        self.cow_gps_list1 = g_list1
        self.cow_gps_list2 = g_list2

    """
        gpsの各時刻での距離を求める
        g_list1 :   gpslist
        g_list2 :   gpslist
    """
    def make_distance_data(self):
        g_list1 = self.cow_gps_list1
        g_list2 = self.cow_gps_list2
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
                i2 += 1
            else:
                i1 += 1
        return distance_list

    """
        gpsの各時刻での被接近度を求める
        g_list1 :   gpslist (main)
        g_list2 :   gpslist (other)
        listのインデックス順にたどっているのでiとi + 1の関係が時間軸では必ずしも直近とは限らないので注意
        dfがごく短い時間 (例えば10分) 間隔でつくられることを想定して，データが欠けまくっていることはないだろう (あっても1回くらい) と考えてこのようにしている
    """
    def make_cosine_data(self):
        g_list1 = self.cow_gps_list1
        g_list2 = self.cow_gps_list2
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

    """
        2頭間の距離がある閾値以内であった回数を取得 (そこから秒数に直すこともできる)
        threshold : 距離の閾値
    """
    def count_near_distance_time(self, threshold):
        g_list1 = self.cow_gps_list1
        g_list2 = self.cow_gps_list2
        count = 0 #カウント用変数
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
                if(d < threshold):
                    count += 1
            elif(t > g_list2[i2].get_datetime()):
                i2 += 1
            else:
                i1 += 1
        return count


class CowsRelation:
    main_cow_id:int #関係を見る対象の牛の個体番号
    main_cow_data:gpslist.GpsNmeaDataList ##関係を見る対象の牛のGPSデータ
    com_members:list #コミュニティメンバのリスト(int) 
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
    def __init__(self, id, data, teams, df):
        self.main_cow_id = id
        self.main_cow_data = data
        self.com_members = teams
        self.cow_data = df

    """
        評価値 (被接近度) を計算する
        評価値は各時刻での被接近度と過去のコミュニティメンバや平均距離から恣意的 (ある思惑によって) に決定した重みによって定量化される
    """
    def calc_evaluation_value(self):
        members = [] #コミュニティメンバのID (一応順番に並べたいので...)
        distances = [] #コミュニティメンバとの距離のリスト　
        cosines = [] #コミュニティメンバとの被接近度のリスト
        for i in range(len(self.cow_data.columns)):
            if(self.cow_data.iloc[0, i] in self.com_members):
                tcr = TwoCowsRelation(self.main_cow_data, self.cow_data.iloc[1, i])
                distance = tcr.make_distance_data()
                d = sum(distance) / len(distance) #平均距離
                cosine = tcr.make_cosine_data()
                c = sum(cosine) / len(cosine) #被接近度の平均
                members.append(self.cow_data.iloc[0, i])
                distances.append(d)
                cosines.append(c)
        return

    """
        重みの決定を行う
        d_list :    距離のリスト
        n_list :    コミュニティ継続度のリスト
        coef   :    それぞれの指標に対する重みへの考慮度を表す係数
    """
    def calc_weight(self, d_list, n_list, coef):
        weights = [] #コミュニティメンバの重みのリスト
        sum_d = 0.0
        sum_n = 0.0
        for d, n in zip(d_list, n_list):
            sum_d += -100 * (1 + math.exp(-1 * (d - 5) / 1 - 1.5)) + 100
            sum_n += n * 5
        for d, n in zip(d_list, n_list):
            wd = -100 * (1 + math.exp(-1 * (d - 5) / 1 - 1.5)) + 100
            wn = n * 5
            weights.append(coef * (wd / sum_d) + (1 - coef) * (wn / sum_n))
        return weights
