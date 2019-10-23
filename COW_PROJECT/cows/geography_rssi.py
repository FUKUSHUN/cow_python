#-*- encoding:utf-8 -*-
import math
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import rssi.rssi_data as rssidata #自作クラス
import rssi.rssi_data_list as rssilist #自作クラス


def get_absolute_speed(p1:rssidata.RSSIData, p2:rssidata.RSSIData):
    """ 速度 (速さ, 角度) を単位 [m/s] で求める
        ここでの速度はGPSの任意の2つのデータ間の平均速度である
        p1	:RSSIData (before)
        p2	:RSSIData (after) """
    lat1, lon1 = p1.get_rssi_info(p1.get_datetime())
    lat2, lon2 = p2.get_rssi_info(p2.get_datetime())
    d, a = get_distance_and_direction(lat1, lon1, lat2, lon2, False)
    return d / (p2.get_datetime() - p1.get_datetime()).total_seconds(), a


def get_cos_sim(p1_after:rssidata.RSSIData, p2_after:rssidata.RSSIData, p1:rssidata.RSSIData, p2:rssidata.RSSIData):
    """ 被接近度 (コサイン類似度) を求める
        p1	:GpsNmeaData (main)
        p2	:GpsNmeaData (other)
        p2_beforeからp2への移動の方向とp2_beforeからp1_beforeへの移動の方向を見たい """
    lat1_a, lon1_a = p1_after.get_rssi_info(p1_after.get_datetime())
    lat1, lon1 = p1.get_rssi_info(p1.get_datetime())
    lat2_a, lon2_a = p2_after.get_rssi_info(p2_after.get_datetime())
    lat2, lon2 = p2.get_rssi_info(p2.get_datetime())

    _, p1deg = get_distance_and_direction(lat1, lon1, lat1_a, lon1_a, True)
    _, p2deg = get_distance_and_direction(lat2, lon2, lat2_a, lon2_a, True)
    _, xdeg = get_distance_and_direction(lat2, lon2, lat1, lon1, True)

    """
    #p2が止まっているときでもp1で代用 (休息中の牛が対象牛に興味を示すという矛盾を抱えているのでペンディング)
    if(p1deg == -1 and xdeg != -1):
        return -1 * math.cos(get_relative_angle(xdeg, p1deg))
    下のifに影響を及ぼす
    """
    #p1, p1_before p2, p2_beforeが同じ座標の時 (どちらの牛も止まっている)
    if(p2deg == -1):
        return 0
    #2頭の位置が同じときにはお互いの速度ベクトルのコサイン類似度で代用
    elif(xdeg == -1 and p1deg != -1 and p2deg != -1):
        return math.cos(get_relative_angle(p1deg, p2deg))
    else:
        #角度がおかしな値を出しているとき
        if(p2deg > 360 or p2deg < 0 or xdeg > 360 or xdeg < 0):
            return 0
        else:
            return math.cos(get_relative_angle(xdeg, p2deg))

		
def get_distance_and_direction(lat1, lon1, lat2, lon2, is_translated):
    """ 距離と角度を返す[m], [度]
        Parameters
        ----------
        lat1, 2		:float	:緯度 (1:before or main, 2:after or other) 
        lon1, 2    	: float:経度 (1:before or main, 2:after or other) 
        is_translated	:boolean (True:ddd.mmmmmの形になっているとき，False:dddmm.mmの形になっているとき)
        角度は北を0度，時計回りを正の方向として度数法 (0 - 360) の値を返す
        参考URL：https://keisan.casio.jp/exec/system/1257670779
        (方位についてはサイトの1/21時点の表記atan2(y, x)をatan2(x, y)に変換して計算している) """	
    latt1, longi1 = lat1, lon1
    latt2, longi2 = lat2, lon2
    #同一座標の時
    if lon1 == lon2 and lat1 == lat2:
        return 0, -1
    else:
        #赤道半径 [m] (地球を球面と仮定して計算している)
        r = 6378137
        #弧度法 ([rad]) に変換
        if(not(is_translated)):
            lat1, lon1 = translate(lat1, lon1)
            lat2, lon2 = translate(lat2, lon2)
        lat1, lon1 = deg_to_rad(lat1, lon1)
        lat2, lon2 = deg_to_rad(lat2, lon2)
        
        try:
            d = r * math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
        except ValueError:
            print("2地点がほとんど同一です", latt1, longi1, latt2, longi2)
            d = r * math.acos(1.0) #誤差で1を超える場合がある (結果的に距離は0になる)
        if math.isnan(d):
            print("距離がNanです")
            d = 0
        a = 90 - math.degrees(math.atan2(math.cos(lat1) * math.tan(lat2) - math.sin(lat1) * math.cos(lon2 - lon1), math.sin(lon2 - lon1)))
        if a < 0:
            a = 360 + a #a will become larger than 0 and smaller than 360
        return d, a
		

def get_relative_angle(a, b):
    """ bからみたaの角度を見る (弧度法 [rad]で返す) """
    if a < 0 or 360 < a:
        #print("角度が不正です")
        return -1
    if b < 0 or 360 < b:
        #print("角度が不正です")
        return -1
    return math.radians(abs(a - b)) # the return x will become 0 < x < 2PI
	

def translate(lat, lon):
    """ 経度・緯度をdddmm.mmmm表記をddd.dddd形式に変換する (他のファイルから呼び出されることもあり) """
    lat_d = lat // 100
    lon_d = lon // 100
    lat_m = (lat % 100) / 60
    lon_m = (lon % 100) / 60
    return lat_d + lat_m, lon_d + lon_m


def deg_to_rad(lat, lon):
    """ ddd.dddd度で表される度数法を弧度法に変換する """
    lat = math.radians(lat)
    lon = math.radians(lon)
    return lat, lon