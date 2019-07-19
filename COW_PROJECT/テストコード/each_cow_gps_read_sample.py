#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import re
import datetime
import cows.cow as Cow
import gc
import cows.geography as geo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import statistics
import cows.momentum_analysys as ma
import numpy as np

def read_gps(cow_id, start, end):

	time_list = []
	distance_list = []
	velocity_list = []
	angle_list = []
	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		t_list = []
		dis_list = []
		vel_list = []
		ang_list = []
		cow = Cow.Cow(cow_id, dt)
		dt = dt + datetime.timedelta(days = 1)
		while(a < dt and a < end):
			gps_list = cow.get_gps_list(a, a + datetime.timedelta(minutes = 60))
			g_before = None
			for i in range(int(len(gps_list) / 5)):
				g = gps_list[i * 5]
				if g_before is not None:
					lat1, lon1, vel1 = g_before.get_gps_info(g_before.get_datetime())
					lat2, lon2, vel2 = g.get_gps_info(g.get_datetime())
					distance, angle = geo.get_distance_and_direction(lat1, lon1, lat2, lon2)
					#print(g.get_datetime().strftime("%Y/%m/%d %H:%M:%S") + " : ", lat2 , ",", lon2)
					t_list.append(g.get_datetime()) #時間の格納
					dis_list.append(distance) #距離の格納
					vel_list.append(vel2) #速さの格納
					ang_list.append(angle) #角度の格納
				g_before = g
			a = a + datetime.timedelta(minutes = 60)
			del gps_list
			gc.collect()
		time_list.append(t_list) #1日分の時間のリストの格納
		distance_list.append(dis_list) #1日分の距離のリストの格納
		velocity_list.append(vel_list) #1日分の速さのリストの格納
		angle_list.append(ang_list) #1日分の角度のリストの格納
		del cow
		gc.collect()
		a = dt
	return time_list, distance_list, velocity_list, angle_list

#プロットする
def plot_velocity_data(t_list, d_list, v_list):
	knot = 0.514444 # 1 knot = 0.51444 m/s
	#日本時間に直した上で牛舎内にいる時間を除いてグラフ化する
	time_tmp_list = []
	distance_tmp_list = []
	velocity_tmp_list = []
	for (t, d, v) in zip(t_list, d_list, v_list):
		t = t + datetime.timedelta(hours = 9)
		if t.hour < 9 or 12 < t.hour:
			time_tmp_list.append(t)
			distance_tmp_list.append(d / 5)
			velocity_tmp_list.append(v * knot)
	t_list = time_tmp_list
	d_list = distance_tmp_list
	v_list = velocity_tmp_list
	
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1) #4行1列の図
	#ax1の設定
	ax1.set_xticklabels(t_list, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
	ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
	ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H-%M")) #日付の表示形式を決定
	#ax1.set_xlim(start, end) #表示範囲を設定
	#ax1.set_ylim(-0.05, 0.05) #表示範囲を設定
	ax1.plot(t_list, v_list, 'b')
	ax1.legend(("Velocity",), loc='upper left')
	#ax2 = ax1.twinx() #グラフを重ねる
	#ax2.plot(t_list, d_list, 'r')
	#ax2.legend(("Distance",), loc='upper left')
	#plt.savefig(t_list[0].strftime("%y-%m-%d") + "-test.png")
	plt.show()

if __name__ == '__main__':
	fc = 0.1 #カットオフ周波数
	start = datetime.datetime(2018, 2, 15, 0, 0, 0)
	end = datetime.datetime(2018, 2, 16, 0, 0, 0)
	time_list, distance_list, velocity_list, angle_list = read_gps(20283, start, end) #2次元リスト (1日分 * 日数分)
	for (t_list, d_list, v_list) in zip(time_list, distance_list, velocity_list):
		print(len(t_list))
		t_list, d_list, v_list = ma.convo_per_minutes(t_list, d_list, v_list)
		#print(len(t_list))
		#t_list, d_list, v_list = ma.mean_per_minutes(t_list, d_list, v_list)
		#print(len(t_list))
		#t_list, d_list, v_list = ma.median_for_minutes(t_list, d_list, v_list)
		print(len(t_list))
		t_list, d_list, v_list =  ma.differ_filter(t_list, d_list, v_list)
		#print(len(t_list))
		#t_list, d_list, v_list =  ma.differ_filter2(t_list, d_list, v_list)
		print(len(t_list))
		plot_velocity_data(t_list, d_list, v_list)
	
