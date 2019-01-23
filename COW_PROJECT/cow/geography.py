#-*- encoding:utf-8 -*-
import math

"""
   Parameters
   ----------
   lat1, 2		:float	:緯度
   lon1, 2    	: float:経度
   どちらも度数法34.xxx, 134.xxxで渡すこと
   角度は北を0度，時計回りを正の方向として度数法 (0 - 360) の値を返す
   参考URL：https://keisan.casio.jp/exec/system/1257670779
    (方位についてはサイトの1/21時点の表記atan2(y, x)をatan2(x, y)に変換して計算している)
"""			
#距離と角度を返す[m], [度]
def get_distance_and_direction(lat1, lon1, lat2, lon2):
	#同一座標の時
	if lon1 == lon2 and lat1 == lat2:
		return 0, -1
	else:
		#赤道半径 [m] (地球を球面と仮定して計算している)
		r = 6378137
		#弧度法 ([rad]) に変換
		lat1, lon1 = translate_rad(lat1, lon1)
		lat2, lon2 = translate_rad(lat2, lon2)
		
		d = r * math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
		if math.isnan(d):
			print("距離がNanです")
			d = 0
		a = 90 - math.degrees(math.atan2(math.cos(lat1) * math.tan(lat2) - math.sin(lat1) * math.cos(lon2 - lon1), math.sin(lon2 - lon1)))
		if a < 0:
			a = 360 + a
		return d, a
		
#bからみたaの角度を見る (弧度法 [rad]で返す)
def get_relative_angle(a, b):
	if a < 0 or 360 < a:
		print("角度が不正です")
		return -1
	if b < 0 or 360 < a:
		print("角度が不正です")
		return -1
	return math.radians(abs(a - b))
	
#経度・緯度をdddmm.mmmm表記をddd.dddd形式に変換し，弧度法 ([rad]) に変換する
def translate_rad(lat, lon):
	lat_d = lat // 100
	lon_d = lon // 100
	lat_m = (lat % 100) / 60
	lon_m = (lon % 100) / 60
	lat = math.radians(lat_d + lat_m)
	lon = math.radians(lon_d + lon_m)
	return lat, lon