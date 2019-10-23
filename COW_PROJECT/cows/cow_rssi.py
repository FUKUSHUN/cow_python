#-*- encoding:utf-8 -*-
import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import rssi.rssi_data_list as rssilist #自作クラス

class Cow:
	cow_id:int
	rssi_list:rssilist.RSSIDataList
	
	def __init__(self, cow_id:int, date:datetime):
		""" 牛の個体番号とその日の位置情報のデータの読み込みを行う
			Parameter
				cow_id	: 読み込むデータのキーとなる牛の個体番号
				date	: 読み込む日付 """
		self.cow_id = cow_id
		self.rssi_list = rssilist.RSSIDataList(date, self.cow_id)
		
	def get_cow_id(self):
		""" 牛の個体番号を取得する """
		return self.cow_id
	
	def get_rssi_list(self, start:datetime, end:datetime):
		""" 位置情報のリストを取得する. 1日の範囲内である必要がある
			Parameter
				start	: 読み込み開始時刻
				end		: 読み込み終了時刻 """
		list = self.rssi_list.get_sub_list(start, end)
		return list
		
	#テスト用 (gps_listを返す)
	def get_test_list(self):
		return self.rssi_list.get_rssi_list()