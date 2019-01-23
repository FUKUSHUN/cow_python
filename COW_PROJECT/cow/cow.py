#-*- encoding:utf-8 -*-
import datetime
import cow.gps.gps_nmea_data_list as gpslist

class Cow:
	cow_id:int
	gps_list:gpslist.GpsNmeaDataList
	
	def __init__(self, cow_id:int, date:datetime):
		self.cow_id = cow_id
		self.gps_list = gpslist.GpsNmeaDataList(date, self.cow_id)
		
	def get_cow_id(self):
		return self.cow_id
	
	def get_gps_list(self, start:datetime, end:datetime):
		list = self.gps_list.get_sub_list(start, end)
		return list
		
	#テスト用 (gps_listを返す)
	def get_test_list(self):
		return self.gps_list.get_gps_list()