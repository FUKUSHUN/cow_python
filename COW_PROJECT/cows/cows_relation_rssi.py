#-*- encoding:utf-8 -*-
import pandas as pd
import datetime
import pickle
import sys
import os
import shutil
import math
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cow_rssi #自作クラス
import cowshed_rssi #自作クラス
import community_history # 自作クラス
import geography_rssi #自作クラス
import rssi.rssi_data_list as rssilist#自作クラス
import rssi.rssi_data as rssidata#自作クラス

class TwoCowsRelation:
    cow_pos_list1:rssilist.RSSIDataList
    cow_pos_list2:rssilist.RSSIDataList
    """
        2頭間の関係を見るクラスで二つのrssi_listが必要
        CowsRelationの中でも2頭間の関係 (コミュニティ生成などで外部から呼び出されることもある. だからクラスを分けた)
    """
    def __init__(self, p_list1, p_list2):
        self.cow_pos_list1 = p_list1
        self.cow_pos_list2 = p_list2

    
    def make_distance_data(self):
        """ rssiの各時刻での距離を求める
            Parameter
                p_list1 :   rssilist
                p_list2 :   rssilist """
        distance_list = []
        i1 = 0 #p_list2のインデックス
        i2 = 0 #p_list2のインデックス
        while(i1 < len(self.cow_pos_list1) and i2 < len(self.cow_pos_list2)):
            t = self.cow_pos_list1[i1].get_datetime()
            if(t == self.cow_pos_list2[i2].get_datetime()):
                lat1, lon1 = self.cow_pos_list1[i1].get_rssi_info(t)
                lat2, lon2 = self.cow_pos_list2[i2].get_rssi_info(self.cow_pos_list2[i2].get_datetime())
                d, _ = geography_rssi.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
                i1 += 1
                i2 += 1
                distance_list.append(d) #左に時刻，右に距離
            elif(t > self.cow_pos_list2[i2].get_datetime()):
                i2 += 1
            else:
                i1 += 1
        return sum(distance_list) / len(distance_list)  # 最後に平均を求めているのはリストとして返すことも想定しているため

    
    def make_cosine_data(self):
        """ rssiの各時刻での被接近度を求める
            Parameter
                p_list1 :   rssilist (main)
                p_list2 :   rssilist (other)
            listのインデックス順にたどっているのでiとi + 1の関係が時間軸では必ずしも直近とは限らないので注意
            dfがごく短い時間 (例えば10分) 間隔でつくられることを想定して，データが欠けまくっていることはないだろう (あっても1回くらい) と考えてこのようにしている """
        cosine_list = []
        i1 = 0 #p_list2のインデックス
        i2 = 0 #p_list2のインデックス
        while(i1 < len(self.cow_pos_list1) and i2 < len(self.cow_pos_list2)):
            t = self.cow_pos_list1[i1].get_datetime()
            if(t == self.cow_pos_list2[i2].get_datetime()):
                if(i1 != len(self.cow_pos_list1) - 1 and i2 != len(self.cow_pos_list2) - 1):
                    cos = geography_rssi.get_cos_sim(self.cow_pos_list1[i1 + 1], self.cow_pos_list2[i2 + 1], self.cow_pos_list1[i1], self.cow_pos_list2[i2])
                    i1 += 1
                    i2 += 1
                    cosine_list.append(cos) #左に時刻，右にコサイン類似度
                else:
                    i1 += 1
                    i2 += 1
            elif(t > self.cow_pos_list2[i2].get_datetime()):
                i2 = i2 + 1
            else:
                i1 = i1 + 1
        return sum(cosine_list) / len(cosine_list) # 最後に平均を求めているのはリストとして返すことも想定しているため

    
    def count_history_score(self, history, cow_id:int):
        """ 過去のコミュニティに牛がいた回数を返す
            Parameter
                history :   コミュニティ履歴    : community_history.CommunityMembers
                cow_id :    探索対象の牛の個体番号 """
        count = 0
        for one in history:
            members = one.get_community_members()
            if(cow_id in members):
                count += 1
        return count

    
    def count_near_distance_time(self, threshold):
        """ 2頭間の距離がある閾値以内であった回数を取得 (そこから秒数に直すこともできる)
            Parameter
                threshold : 距離の閾値 """
        count = 0 #カウント用変数
        i1 = 0 #p_list2のインデックス
        i2 = 0 #p_list2のインデックス
        while(i1 < len(self.cow_pos_list1) and i2 < len(self.cow_pos_list2)):
            t = self.cow_pos_list1[i1].get_datetime()
            if(t == self.cow_pos_list2[i2].get_datetime()):
                lat1, lon1 = self.cow_pos_list1[i1].get_rssi_info(t)
                lat2, lon2 = self.cow_pos_list2[i2].get_rssi_info(self.cow_pos_list2[i2].get_datetime())
                d, _ = geography_rssi.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
                i1 += 1
                i2 += 1
                if(d < threshold):
                    count += 1
            elif(t > self.cow_pos_list2[i2].get_datetime()):
                i2 += 1
            else:
                i1 += 1
        return count


class CowsRelation:
    __picpath = "serial/"
    __time_thought_as_history = 180 # コミュニティ履歴の参考時間 (分)
    time:datetime.datetime # 時刻
    main_cow_id:int #関係を見る対象の牛の個体番号
    main_cow_data:rssilist.RSSIDataList ##関係を見る対象の牛のRSSIデータ
    com_members:community_history.CommunityMembers #コミュニティメンバのリスト(CommunityMembers) 
    member_history:list #過去のコミュニティメンバのリスト(CommunityMembers) (シリアル化して保存)
    cow_data:pd.DataFrame #牛のリスト (現時点ではpandasのデータフレーム型でidと位置情報履歴を保持, main_cowも含む) 

    """
        dfの形式はcowshed_rssi.Cowshed.get_cow_list()を参照すること
        ------------------------
        ID      | 20... | 20...
        ------------------------
        Data    | [...] | [...]
        ------------------------
        の形をしている (1/29時点)
        行と列を逆にする方が良いかもしれない. 速さとアクセスのしやすさのトレードオフ？
    """
    def __init__(self, id:int, dt, data, teams:list, df:pd.DataFrame):
        self.main_cow_id = id
        self.time = dt
        self.main_cow_data = data
        members = [c for c in teams if c != self.main_cow_id] #自分を除外
        self.com_members = community_history.CommunityMembers(self.time, members)
        self.cow_data = df
        self.member_history = [] # 空のリストを用意
        self.reset_member_history() # 12時ならコミュニティ履歴をリセット
        self.unpickling()
        self.pickling()

    
    def calc_evaluation_value(self):
        """ 評価値 (被接近度) を計算する
            評価値は各時刻での被接近度と過去のコミュニティメンバや平均距離から恣意的 (ある思惑によって) に決定した重みによって定量化される """
        members = [] #コミュニティメンバのID (一応順番に並べたいので...)
        distances = [] #コミュニティメンバとの距離のリスト
        histories = [] #コミュニティ継続度のリスト
        cosines = [] #コミュニティメンバとの被接近度のリスト
        for i in range(len(self.cow_data.columns)):
            members = self.com_members.get_community_members()
            if(self.cow_data.iloc[0, i] in members):
                tcr = TwoCowsRelation(self.main_cow_data, self.cow_data.iloc[1, i]) # 2頭のRssiDataList
                distance = tcr.make_distance_data()
                number = tcr.count_history_score(self.member_history, self.cow_data.iloc[0, i]) # コミュニティ履歴と個体番号
                cosine = tcr.make_cosine_data()
                members.append(self.cow_data.iloc[0, i])
                distances.append(distance)
                histories.append(number)
                cosines.append(cosine)
        weights = self.calc_weight(distances, histories, 0.5)
        value = 0.0
        if (weights is not None):
            for c, w in zip(cosines, weights):
                value += c * w
        return value

    
    def calc_evaluation_value2(self):
        """ 評価値 (被接近度) を計算する
            評価値は被接近度が最大となる牛の被接近度を採用する """
        cosines = [] #コミュニティメンバとの被接近度のリスト
        for i in range(len(self.cow_data.columns)):
            members = self.com_members.get_community_members()
            if(self.cow_data.iloc[0, i] in members):
                tcr = TwoCowsRelation(self.main_cow_data, self.cow_data.iloc[1, i]) # 2頭のRssiDataList
                cosine = tcr.make_cosine_data()
                cosines.append(cosine)
        value = 0.0
        for c in cosines:
            if(abs(value) < abs(c)):
                value = c
        return value

    
    def calc_weight(self, d_list, n_list, coef):
        """ 重みの決定を行う
            Parameter
                d_list :    距離のリスト
                n_list :    コミュニティ継続度のリスト
                coef   :    それぞれの指標に対する重みへの考慮度を表す係数 """
        weights = [] #コミュニティメンバの重みのリスト
        sum_d = 0.0
        sum_n = 0.0
        for d, n in zip(d_list, n_list):
            sum_d += -1 / (1 + math.exp(-1 * (d - 5) / 1 - 1.5)) + 1
            sum_n += 1 / (1 + math.exp(-1 * (n - 5) / 1))
        for d, n in zip(d_list, n_list):
            wd = -1 / (1 + math.exp(-1 * (d - 5) / 1 - 1.5)) + 1
            wn = 1 / (1 + math.exp(-1 * (n - 5) / 1))
            if (sum_d == 0 and sum_n == 0):
                return None # どちらも0なら評価値の算出は不可能
            elif(sum_d == 0):
                weights.append(wn / sum_n) # 距離の総和が0ならコミュニティ履歴から重みを算出
            elif(sum_n == 0):
                weights.append(wd / sum_d) # コミュニティ履歴がないなら距離から重みを算出
            else:
                weights.append(coef * (wd / sum_d) + (1 - coef) * (wn / sum_n))
        return weights

    
    def pickling(self):
        """ シリアル化するファイルに過去18回分のコミュニティ履歴を書き込む """
        filepath = self.__picpath + str(self.main_cow_id) + "_rssi.pickle"
        temp_history = self.member_history # 現在のコミュニティ履歴
        with open(filepath, mode = "wb") as f:
            temp_history.append(self.com_members) #リストの一番最後に追加
            pickle.dump(temp_history, f) # 漬物にする
    
    
    def unpickling(self):
        """ シリアライズファイルからコミュニティ履歴を見る """
        filepath = self.__picpath + str(self.main_cow_id) + "_rssi.pickle"
        if(os.path.exists(filepath)):
            with open(filepath, mode = "rb") as f:
                self.member_history = pickle.load(f)
                self.set_member_history(self.member_history) # 考慮するコミュニティ履歴をセット
        return


    def set_member_history(self, history_data):
        """ 時間内にあるコミュニティ履歴を取得する. unpickleでコミュニティ履歴全体はロード済みとする
            Parameter
                history_data    : コミュニティ履歴のリスト """
        history = []
        time1 = self.time - datetime.timedelta(minutes=self.__time_thought_as_history) # この時間から現在の1つ前までの履歴を参照する
        for members in history_data:
            if (members.confirm_if_inside_times(time1, self.time)):
                history.append(members)
        self.member_history = history
        return
    

    def reset_member_history(self):
        """ 毎日の12時にコミュニティ履歴をリセット """
        if (self.time.hour == 12 and self.time.minute == 0):
            #シリアライズファイルの初期化
            shutil.rmtree("serial")
            os.mkdir("serial")
