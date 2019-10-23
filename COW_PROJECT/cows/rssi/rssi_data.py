#-*- encoding:utf-8 -*-
import datetime

class RSSIData:
	dt:datetime.datetime #datetime型 (GMT)
	latitude = None
	longitude = None
	
	def __init__(self, dt, lat, lon):
		self.dt = dt
		self.latitude = lat
		self.longitude = lon


	def get_rssi_info(self, dt):
		if self.dt == dt:
			return self.latitude, self.longitude
		else:
			print("データなし: " + dt.strftime("%Y/%m/%d %H:%M:%S"))
			return None, None
			
	def get_datetime(self):
		return self.dt #datetime型