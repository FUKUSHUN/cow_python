#-*- encoding:utf-8 -*-
import datetime

class GpsNmeaData:
	dt:datetime.datetime #datetime型 (GMT)
	latitude = None
	longitude = None
	velocity = None #[knot]
	
	def __init__(self, dt, lat, lon, vel):
		self.dt = dt
		self.latitude = lat
		self.longitude = lon
		self.velocity = vel

	def get_gps_info(self, dt):
		if self.dt == dt:
			return self.latitude, self.longitude, self.velocity
		else:
			print(dt.strftime("%Y/%m/%d %H:%M:%S") + "," + dt.strftime("%Y/%m/%d %H:%M:%S"))
			return None, None, None
			
	def get_datetime(self):
		return self.dt #datetime型

	
	def set_datetime(self, dt):
		self.dt = dt # datetime型（時差を直したい時など）