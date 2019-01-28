#-*- encoding:utf-8 -*-
import datetime
import sqlite3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import gps_nmea_data as gps #自作クラス

class GpsNmeaDataList:
	_error_threshold = 0.15 #[m/s] 固定, 動いたかどうかを判断する速さの閾値
	_db_file_path = "../CowTagOutput/DB/PosDB/"
	gps_list = [] #gpsのリスト
	
	def __init__(self, date:datetime, cow_id):
		self.gps_list = []
		self.read_from_database(date, cow_id, False)
		
	#gpsのリストを取得する
	def get_gps_list(self):
		return self.gps_list
		
	#gpsのリストのサブリストを取得する (時間順に並んでいることが想定されている)
	"""
	return------
		list	: gpsのデータリスト (このクラスではないので注意)
	"""
	def get_sub_list(self, start:datetime, end:datetime):
		sublist = []
		for g in self.gps_list:
			if end < g.get_datetime():
				break
			elif start <= g.get_datetime() and g.get_datetime() < end:
				sublist.append(g)
		return sublist
		
	#最初にデータベースからリストを作成する (1日ごと)
	"""
	parameter------
		linear_flag	: 線形補間するかどうか
	"""
	def read_from_database(self, date:datetime, cow_id:int, linear_flag):
		start = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
		end = start + datetime.timedelta(days = 1)
		filename = self._db_file_path + self.__make_file_format(start) + ".db"
		conn = sqlite3.connect(filename)
		c = conn.cursor()
		sql = "SELECT * FROM `" + str(cow_id) + "` WHERE time >= " + "'" + start.strftime("%Y/%m/%d %H:%M:%S") + "'" + "AND time < " + "'" + end.strftime("%Y/%m/%d %H:%M:%S") + "'" #SQLの文章
		itr = c.execute(sql) #SQL文の実行
		g_before = None
		for row in itr:
			dt = datetime.datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S")
			g = gps.GpsNmeaData(dt, row[1], row[2], row[3])
			if g_before is not None:
				#速さが閾値self._error_threshold以下の時は前回の場所を踏襲
				if (float(row[3]) * 1852 / 3600) < self._error_threshold:
					g_list = self.__normal_interpolation(g_before, g)
					for g_i in g_list:
						self.gps_list.append(g_i)
					lat, lon, _ = g_before.get_gps_info(g_before.get_datetime())
					g = gps.GpsNmeaData(dt, lat, lon, row[3])
				#普通の線形補間
				elif linear_flag:
					g_list = self.__linear_interpolation(g_before, g)
					for g_i in g_list:
						self.gps_list.append(g_i)
				#線形補間なし (他の牛との時間合わせのために1sごとに出力する)
				else:
					g_list = self.__normal_interpolation(g_before, g)
					for g_i in g_list:
						self.gps_list.append(g_i)
			else:
				self.gps_list.append(g)
			g_before = g
		conn.close()
		return
		
	#1s間隔のデータに直すために線形補間を行う (緯度・経度の線形補間であることを留意)
	def __linear_interpolation(self, start:gps.GpsNmeaData, end:gps.GpsNmeaData):
		dt_s = start.get_datetime()
		dt_e = end.get_datetime()
		interval = int((dt_e - dt_s).total_seconds()) #秒単位で時間の差分をとる
		interpolation_list = []
		lat_s, lon_s, vel_s = start.get_gps_info(dt_s)
		lat_e, lon_e, vel_e = end.get_gps_info(dt_e)
		lat_interval = (lat_e - lat_s) / interval
		lon_interval = (lon_e - lon_s) / interval
		vel_interval = (vel_e - vel_s) / interval
		for i in range(interval):
			lat = lat_s + lat_interval * (i)
			lon = lon_s + lon_interval * (i)
			vel = vel_s + vel_interval * (i)
			#print(lat, ",", lon, ",", vel)
			dt_i = dt_s + datetime.timedelta(seconds = i)
			g = gps.GpsNmeaData(dt_i, lat, lon, vel)
			interpolation_list.append(g)
		return interpolation_list #startはこの補間リストに含まれない (前回の補間時に含まれるため重複を避けたい...)
	
	#1s間隔のデータに直すために間をstartと同じ値で埋め合わせる (個体ごとの観測のズレをなくしたい)
	def __normal_interpolation(self, start:gps.GpsNmeaData, end:gps.GpsNmeaData):
		dt_s = start.get_datetime()
		dt_e = end.get_datetime()
		interval = int((dt_e - dt_s).total_seconds()) #秒単位で時間の差分をとる
		interpolation_list = []
		lat_s, lon_s, vel_s = start.get_gps_info(dt_s)
		for i in range(interval):
			dt_i = dt_s + datetime.timedelta(seconds = i)
			g = gps.GpsNmeaData(dt_i, lat_s, lon_s, vel_s)
			interpolation_list.append(g)
		return interpolation_list #startはこの補間リストに含まれない (前回の補間時に含まれるため重複を避けたい...)
		
	#yymmdd.dbの形式にする (dbのファイルネーム規則)
	def __make_file_format(self, dt):
		y = dt.year % 100 #20XXのXXの部分だけ取り出す
		m = dt.month
		d = dt.day
		return str(y * 10000 + m * 100 + d)
		