#-*- encoding:utf-8 -*-
import pandas as pd
import pickle
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
        CowsRelationの分社化みたいなもの (コミュニティ生成などで外部から呼び出されることもある. だから分社化した)
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
        distance_list = []
        i1 = 0 #g_list2のインデックス
        i2 = 0 #g_list2のインデックス
        while(i1 < len(self.cow_gps_list1) and i2 < len(self.cow_gps_list2)):
            t = self.cow_gps_list1[i1].get_datetime()
            if(t == self.cow_gps_list2[i2].get_datetime()):
                lat1, lon1, _ = self.cow_gps_list1[i1].get_gps_info(t)
                lat2, lon2, _ = self.cow_gps_list2[i2].get_gps_info(self.cow_gps_list2[i2].get_datetime())
                d, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2)
                i1 += 1
                i2 += 1
                distance_list.append(d) #左に時刻，右に距離
            elif(t > self.cow_gps_list2[i2].get_datetime()):
                i2 += 1
            else:
                i1 += 1
        return sum(distance_list) / len(distance_list)  # 最後に平均を求めているのはリストとして返すことも想定しているため

    """
        gpsの各時刻での被接近度を求める
        g_list1 :   gpslist (main)
        g_list2 :   gpslist (other)
        listのインデックス順にたどっているのでiとi + 1の関係が時間軸では必ずしも直近とは限らないので注意
        dfがごく短い時間 (例えば10分) 間隔でつくられることを想定して，データが欠けまくっていることはないだろう (あっても1回くらい) と考えてこのようにしている
    """
    def make_cosine_data(self):
        cosine_list = []
        i1 = 0 #g_list2のインデックス
        i2 = 0 #g_list2のインデックス
        while(i1 < len(self.cow_gps_list1) and i2 < len(self.cow_gps_list2)):
            t = self.cow_gps_list1[i1].get_datetime()
            if(t == self.cow_gps_list2[i2].get_datetime()):
                if(i1 != len(self.cow_gps_list1) - 1 and i2 != len(self.cow_gps_list2) - 1):
                    cos = geography.get_cos_sim(self.cow_gps_list1[i1 + 1], self.cow_gps_list2[i2 + 1], self.cow_gps_list1[i1], self.cow_gps_list2[i2])
                    i1 += 1
                    i2 += 1
                    cosine_list.append(cos) #左に時刻，右にコサイン類似度
                else:
                    i1 += 1
                    i2 += 1
            elif(t > self.cow_gps_list2[i2].get_datetime()):
                i2 = i2 + 1
            else:
                i1 = i1 + 1
        return sum(cosine_list) / len(cosine_list) # 最後に平均を求めているのはリストとして返すことも想定しているため

    """
        過去のコミュニティに牛がいた回数を返す
        history :   コミュニティ履歴
        cow_id :    探索対象の牛の個体番号
    """
    def count_history_score(self, history:list, cow_id:int):
        count = 0
        for members in history:
            if(cow_id in members):
                count += 1
        return count

    """
        2頭間の距離がある閾値以内であった回数を取得 (そこから秒数に直すこともできる)
        threshold : 距離の閾値
    """
    def count_near_distance_time(self, threshold):
        count = 0 #カウント用変数
        i1 = 0 #g_list2のインデックス
        i2 = 0 #g_list2のインデックス
        while(i1 < len(self.cow_gps_list1) and i2 < len(self.cow_gps_list2)):
            t = self.cow_gps_list1[i1].get_datetime()
            if(t == self.cow_gps_list2[i2].get_datetime()):
                lat1, lon1, _ = self.cow_gps_list1[i1].get_gps_info(t)
                lat2, lon2, _ = self.cow_gps_list2[i2].get_gps_info(self.cow_gps_list2[i2].get_datetime())
                d, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2)
                i1 += 1
                i2 += 1
                if(d < threshold):
                    count += 1
            elif(t > self.cow_gps_list2[i2].get_datetime()):
                i2 += 1
            else:
                i1 += 1
        return count


class CowsRelation:
    __picpath = "serial/"
    main_cow_id:int #関係を見る対象の牛の個体番号
    main_cow_data:gpslist.GpsNmeaDataList ##関係を見る対象の牛のGPSデータ
    com_members:list #コミュニティメンバのリスト(int) 
    member_history:list #過去のコミュニティメンバのリスト (シリアル化して保存)
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
    def __init__(self, id, data, teams:list, df:pd.DataFrame):
        self.main_cow_id = id
        self.main_cow_data = data
        self.com_members = [c for c in teams if c != self.main_cow_id] #自分を除外
        self.cow_data = df
        self.member_history = []
        self.unpickling()
        self.pickling()

    """
        評価値 (被接近度) を計算する
        評価値は各時刻での被接近度と過去のコミュニティメンバや平均距離から恣意的 (ある思惑によって) に決定した重みによって定量化される
    """
    def calc_evaluation_value(self):
        if(len(self.member_history) < 18):
            return None
        else:
            members = [] #コミュニティメンバのID (一応順番に並べたいので...)
            distances = [] #コミュニティメンバとの距離のリスト
            histories = [] #コミュニティ継続度のリスト
            cosines = [] #コミュニティメンバとの被接近度のリスト
            for i in range(len(self.cow_data.columns)):
                if(self.cow_data.iloc[0, i] in self.com_members):
                    tcr = TwoCowsRelation(self.main_cow_data, self.cow_data.iloc[1, i])
                    distance = tcr.make_distance_data()
                    number = tcr.count_history_score(self.member_history, self.cow_data.iloc[0, i])
                    cosine = tcr.make_cosine_data()
                    members.append(self.cow_data.iloc[0, i])
                    distances.append(distance)
                    histories.append(number)
                    cosines.append(cosine)
            weights = self.calc_weight(distances, histories, 0.5)
            value = 0.0
            for c, w in zip(cosines, weights):
                value += c * w
            return value * 5 #線形補完を行っていないときは前回の位置を引きついだ4回の無駄 (値が0) を含むため

    """
        評価値 (被接近度) を計算する
        評価値は被接近度が最大となる牛の被接近度を採用する
    """
    def calc_evaluation_value2(self):
        cosines = [] #コミュニティメンバとの被接近度のリスト
        for i in range(len(self.cow_data.columns)):
            if(self.cow_data.iloc[0, i] in self.com_members):
                tcr = TwoCowsRelation(self.main_cow_data, self.cow_data.iloc[1, i])
                cosine = tcr.make_cosine_data()
                cosines.append(cosine)
        value = 0.0
        for c in cosines:
            if(abs(value) < abs(c)):
                value = c
        return value * 5

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
            sum_d += -1 / (1 + math.exp(-1 * (d - 5) / 1 - 1.5)) + 1
            sum_n += 1 / (1 + math.exp(-1 * (n - 9) / 1.8 - 1.0))
        for d, n in zip(d_list, n_list):
            wd = -1 / (1 + math.exp(-1 * (d - 5) / 1 - 1.5)) + 1
            wn = 1 / (1 + math.exp(-1 * (n - 9) / 1.8 - 1.0))
            if(sum_d == 0):
                weights.append(wn / sum_n)
            elif(sum_n == 0):
                weights.append(wd / sum_d)
            else:
                weights.append(coef * (wd / sum_d) + (1 - coef) * (wn / sum_n))
        return weights

    """
        シリアル化するファイルに過去18回分のコミュニティ履歴を書き込む
    """
    def pickling(self):
        filepath = self.__picpath + str(self.main_cow_id) + ".pickle"
        with open(filepath, mode = "wb") as f:
            if(len(self.member_history) < 18):
                temp_history = self.member_history
                temp_history.append(self.com_members) #リストの一番最後に追加
            elif(len(self.member_history) >= 18):
                temp_history = self.member_history
                temp_history.pop(0) #リストの一番最初を削除
                temp_history.append(self.com_members) #リストの一番最後に追加
            pickle.dump(temp_history, f)
    
    """
        シリアライズファイルからコミュニティ履歴を見る
    """
    def unpickling(self):
        filepath = self.__picpath + str(self.main_cow_id) + ".pickle"
        if(os.path.exists(filepath)):
            with open(filepath, mode = "rb") as f:
                self.member_history = pickle.load(f)
        return
