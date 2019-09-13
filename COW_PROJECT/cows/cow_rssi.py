#-*- encoding:utf-8 -*-
import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import gps.rssi_data_list as rssilist #自作クラス

class Cow:
	cow_id:int
	rssi_list:rssilist.RSSIDataList
	
	def __init__(self, cow_id:int, date:datetime):
		self.cow_id = cow_id
		self.rssi_list = rssilist.RSSIDataList(date, self.cow_id)
		
	def get_cow_id(self):
		return self.cow_id
	
	def get_gps_list(self, start:datetime, end:datetime):
		list = self.rssi_list.get_sub_list(start, end)
		return list
		
	#テスト用 (gps_listを返す)
	def get_test_list(self):
		return self.rssi_list.get_rssi_list()