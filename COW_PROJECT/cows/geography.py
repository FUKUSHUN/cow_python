#-*- encoding:utf-8 -*-
import math
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import gps.gps_nmea_data as gpsdata #自作クラス
import gps.gps_nmea_data_list as gpslist #自作クラス

"""
	速度 (速さ, 角度) を単位 [m/s] で求める
	ここでの速度はGPSの任意の2つのデータ間の平均速度である
	g1	:GpsNmeaData (before)
	g2	:GpsNmeaData (after)
"""
def get_absolute_speed(g1:gpsdata.GpsNmeaData, g2:gpsdata.GpsNmeaData):
	lat1, lon1, _ = g1.get_gps_info(g1.get_datetime())
	lat2, lon2, _ = g2.get_gps_info(g2.get_datetime())
	d, a = get_distance_and_direction(lat1, lon1, lat2, lon2)
	return d / (g2.get_datetime() - g1.get_datetime()).total_seconds(), a

"""
	被接近度 (コサイン類似度) を求める
	g1	:GpsNmeaData (main)
	g2	:GpsNmeaData (other)
	g2_beforeからg2への移動の方向とg2_beforeからg1_beforeへの移動の方向を見たい
	
"""
def get_cos_sim(g1_after:gpsdata.GpsNmeaData, g2_after:gpsdata.GpsNmeaData, g1:gpsdata.GpsNmeaData, g2:gpsdata.GpsNmeaData):
	lat1_a, lon1_a, _ = g1_after.get_gps_info(g1_after.get_datetime())
	lat1, lon1, _ = g1.get_gps_info(g1.get_datetime())
	lat2_a, lon2_a, _ = g2_after.get_gps_info(g2_after.get_datetime())
	lat2, lon2, _ = g2.get_gps_info(g2.get_datetime())

	_, g1deg = get_distance_and_direction(lat1, lon1, lat1_a, lon1_a)
	_, g2deg = get_distance_and_direction(lat2, lon2, lat2_a, lon2_a)
	_, xdeg = get_distance_and_direction(lat2, lon2, lat1, lon1)

	"""
	#g2が止まっているときでもg1で代用 (休息中の牛が対象牛に興味を示すという矛盾を抱えているのでペンディング)
	if(g1deg == -1 and xdeg != -1):
		return -1 * math.cos(get_relative_angle(xdeg, g1deg))
	下のifに影響を及ぼす
	"""
	#g1, g1_before g2, g2_beforeが同じ座標の時 (どちらの牛も止まっている)
	if(g2deg == -1):
		return 0
	#2頭の位置が同じときにはお互いの速度ベクトルのコサイン類似度で代用
	elif(xdeg == -1 and g1deg != -1 and g2deg != -1):
		return math.cos(get_relative_angle(g1deg, g2deg))
	else:
		#角度がおかしな値を出しているとき
		if(g2deg > 360 or g2deg < 0 or xdeg > 360 or xdeg < 0):
			return 0
		else:
			return math.cos(get_relative_angle(xdeg, g2deg))


"""
	距離と角度を返す[m], [度]
   	Parameters
   	----------
   	lat1, 2		:float	:緯度 (1:before or main, 2:after or other) 
   	lon1, 2    	: float:経度 (1:before or main, 2:after or other) 
   	どちらも度数法34.xxx, 134.xxxで渡すこと
   	角度は北を0度，時計回りを正の方向として度数法 (0 - 360) の値を返す
   	参考URL：https://keisan.casio.jp/exec/system/1257670779
    (方位についてはサイトの1/21時点の表記atan2(y, x)をatan2(x, y)に変換して計算している)
"""			
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
		
		try:
			d = r * math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
		except ValueError:
			print(lat1, lon1, lat2, lon2)
			d = r * math.acos(1.0) #誤差で1を超える場合がある (結果的に距離は0になる)
		if math.isnan(d):
			print("距離がNanです")
			d = 0
		a = 90 - math.degrees(math.atan2(math.cos(lat1) * math.tan(lat2) - math.sin(lat1) * math.cos(lon2 - lon1), math.sin(lon2 - lon1)))
		if a < 0:
			a = 360 + a #a will become larger than 0 and smaller than 360
		return d, a
		
#bからみたaの角度を見る (弧度法 [rad]で返す)
def get_relative_angle(a, b):
	if a < 0 or 360 < a:
		#print("角度が不正です")
		return -1
	if b < 0 or 360 < b:
		#print("角度が不正です")
		return -1
	return math.radians(abs(a - b)) # the return x will become 0 < x < 2PI
	
#経度・緯度をdddmm.mmmm表記をddd.dddd形式に変換し，弧度法 ([rad]) に変換する
def translate_rad(lat, lon):
	lat_d = lat // 100
	lon_d = lon // 100
	lat_m = (lat % 100) / 60
	lon_m = (lon % 100) / 60
	lat = math.radians(lat_d + lat_m)
	lon = math.radians(lon_d + lon_m)
	return lat, lon