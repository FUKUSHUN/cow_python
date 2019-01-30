#-*- encoding:utf-8 -*-
import datetime
import csv
import pandas as pd
import re
import time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cow #自作クラス

class Cowshed:
	date:str #YYYY/mm/dd
	cow_list:list #Cow型のリスト
	record_file_path = "../CowTagOutput/csv/"
	"""
		その日いた牛を登録する
		日付をキーにしてコンストラクタでGPSデータの読み込み
	"""
	def __init__(self, date:datetime):
		self.cow_list = []
		self.date = date.strftime("%Y/%m/%d")
		self.record_file_path = self.record_file_path + self.date[:4] + "-" + self.date[5:7] + ".csv"
		self.__read_from_db(self.__get_cow_list())
	
	#1頭ずつデータベースからGPS情報を読み込む
	def __read_from_db(self, cow_id_list):
		dt = datetime.datetime(int(self.date[:4]), int(self.date[5:7]), int(self.date[8:10]))
		print("reading cow information : " + self.date)
		for cow_id in cow_id_list:
			c = cow.Cow(int(cow_id), dt)
			self.cow_list.append(c)
		print("finesh reading cow information : " + self.date)
		
	#csvファイルからその日第一放牧場にいた牛の個体番号のリストを取得する
	def __get_cow_list(self):
		f = open(self.record_file_path, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		cow_id_list = []
		for row in rd:
			if re.match(self.date, row[0]):
				cow_id_list = row[1:]
		f.close()
		return cow_id_list
		
	#牛のリストを取得する (pandasのdataframe型)
	def get_cow_list(self, start:datetime, end:datetime):
		cow_id_list = []
		gps_data_list = []
		for c in self.cow_list:
			#start_time = time.time()
			cow_id_list.append(c.get_cow_id())
			#end_time = time.time()
			gps_data_list.append(c.get_gps_list(start, end))
			#print("{0}".format(end_time - start_time) + " [sec]")
		df = pd.DataFrame([cow_id_list, gps_data_list], index = ["Id", "Data"])
		del c
		"""
		data_list = []
		for c in self.cow_list:
			data = []
			data.append(c.get_cow_id())
			data.append(c.get_gps_list(start, end))
			data_list.append(data)
		df = pd.DataFrame(data_list, columns = ["Id", "Data"], index_col = 0)
		"""
		del cow_id_list
		del gps_data_list
		return df