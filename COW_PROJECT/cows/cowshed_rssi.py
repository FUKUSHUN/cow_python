#-*- encoding:utf-8 -*-
import datetime
import csv
import pandas as pd
import glob
import re
import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cow_rssi #自作クラス

class Cowshed:
    date:str #YYYY/mm/dd
    cow_list:list #Cow型のリスト
    csv_path = "for_web/rssi2latlon/"
    

    def __init__(self, date:datetime):
        """ その日いた牛を登録する
		    日付をキーにしてコンストラクタでRSSIデータの読み込み """
        self.cow_list = []
        self.date = date.strftime("%Y/%m/%d")
        self.csv_path += date.strftime("%Y%m%d") + "/"
        self.__read_from_db(self.__get_cow_id_list())
    
    

    def __read_from_db(self, cow_id_list):
        """ 1頭ずつデータベースからRSSI情報を読み込む """
        dt = datetime.datetime(int(self.date[:4]), int(self.date[5:7]), int(self.date[8:10]))
        print("reading cow information : " + self.date)
        for cow_id in cow_id_list:
            c = cow_rssi.Cow(int(cow_id), dt)
            self.cow_list.append(c)
        print("finished reading cow information : " + self.date)
		
	
    def __get_cow_id_list(self):
        """ csvファイルからその日第一放牧場にいた牛の個体番号のリストを取得する """
        files = glob.glob(self.csv_path + "*.csv")
        cow_id_list = []
        for filepath in files:
            cow_id = os.path.basename(filepath)
            cow_id_list.append(int(cow_id[:5]))
        return cow_id_list
		
	
    def get_cow_list(self, start:datetime, end:datetime):
        """ 牛のリストを取得する (pandasのdataframe型) """
        cow_id_list = []
        rssi_data_list = []
        for c in self.cow_list:
            cow_id_list.append(c.get_cow_id())
            rssi_data_list.append(c.get_rssi_list(start, end))
        df = pd.DataFrame([cow_id_list, rssi_data_list], index = ["Id", "Data"])
        del c
        del cow_id_list
        del rssi_data_list
        return df


    def get_cow_ids(self):
        """ 外部から呼び出し用にその時の牛の個体番号を返す """
        cow_ids = []
        for cow in self.cow_list:
            cow_ids.append(cow.get_cow_id())
        return cow_ids