#-*- encoding:utf-8 -*-
import datetime
import csv
import sqlite3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import gps_nmea_data as gps #自作クラス

class RSSIDataList:
    _csv_file_path = "for_web/rssi2latlon/"
    rssi_list = [] #rssiのリスト
	
    def __init__(self, date:datetime, cow_id):
        self.rssi_list = []
        self.read_from_csv(date, cow_id)

    """	
    rssiのリストを取得する
    """
    def get_rssi_list(self):
        return self.rssi_list
		
    """
    gpsのリストのサブリストを取得する (時間順に並んでいることが想定されている)
    return------
        list	: rasiのデータリスト (このクラスではないので注意)
    """
    def get_sub_list(self, start:datetime, end:datetime):
        sublist = []
        for g in self.rssi_list:
            if end < g.get_datetime():
                break
            elif start <= g.get_datetime() and g.get_datetime() < end:
                sublist.append(g)
        return sublist
    
    """
    CSVからリストを作成する (1日ごと)
    parameter
        date    : 
        cow_id  : 
    """
    def read_from_csv(self, date:datetime, cow_id:int):
        filename = self._csv_file_path + self.__make_file_format(date) + "/" + str(cow_id) + ".csv"
        with open(filename, mode = "r") as f:
            reader = csv.reader(f)
            index = 0
            for row in reader:
                if (index % 5 == 0):
                    time = datetime.datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S")
                    lat = float(row[1])
                    lon = float(row[2])
                    vel = 1 # 適当
                    g = gps.GpsNmeaData(time, lat, lon, vel)
                    self.rssi_list.append(g)
            index += 1
        return

    #yymmdd.dbの形式にする (dbのファイルネーム規則)
    def __make_file_format(self, dt):
        y = dt.year
        m = dt.month
        d = dt.day
        return str(y * 10000 + m * 100 + d)
