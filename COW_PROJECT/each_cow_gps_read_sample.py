#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import re
import datetime
import cow.cow as Cow
import gc
import cow.geography as geo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import statistics

def read_gps():
	start = datetime.datetime(2018, 2, 15, 0, 0, 0)
	end = datetime.datetime(2018, 2, 18, 0, 0, 0)

	time_list = []
	distance_list = []
	velocity_list = []
	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		t_list = []
		dis_list = []
		vel_list = []
		cow = Cow.Cow(20283, dt)
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
				g_before = g
			a = a + datetime.timedelta(minutes = 60)
			del gps_list
			gc.collect()
		time_list.append(t_list) #1日分の時間のリストの格納
		distance_list.append(dis_list) #1日分の距離のリストの格納
		velocity_list.append(vel_list) #1日分の速さのリストの格納
		del cow
		gc.collect()
		a = dt
	return time_list, distance_list, velocity_list 
	
#1分間の平均を求める (5s間隔なので12回分の平均をとるだけ)
def mean_per_minutes(t_list, d_list, v_list):
	count = 0
	new_t_list = []
	new_d_list = []
	new_v_list = []
	sum_d = 0.0
	sum_v = 0.0
	for (t, d, v) in zip(t_list, d_list, v_list):
		count += 1
		sum_d += d
		sum_v += v
		if(count == 12):
			new_t_list.append(datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, 0))
			new_d_list.append(sum_d / count)
			new_v_list.append(sum_v / count)
			count = 0
			sum_d = 0.0
			sum_v = 0.0
	return new_t_list, new_d_list, new_v_list

#1分間の中央値を求める (5s間隔なので12個データの中央値を求めるだけ)
def median_for_minutes(t_list, d_list, v_list):
	count = 0
	new_t_list = []
	new_d_list = []
	new_v_list = []
	sum_d = []
	sum_v = []
	for (t, d, v) in zip(t_list, d_list, v_list):
		count += 1
		sum_d.append(d)
		sum_v.append(v)
		if(count == 12):
			new_t_list.append(datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, 0))
			new_d_list.append(statistics.median(sum_d))
			new_v_list.append(statistics.median(sum_v))
			count = 0
			sum_d = []
			sum_v = []
	return new_t_list, new_d_list, new_v_list
	
#1分間の移動平均を求める (5s間隔なので12回分の平均をずらしなが足し合わせる)
def convo_per_minutes(t_list, d_list, v_list):
	count = 0
	new_t_list = []
	new_d_list = []
	new_v_list = []
	sum_d = []
	sum_v = []
	for (t, d, v) in zip(t_list, d_list, v_list):
		sum_d.append(d)
		sum_v.append(v)
		if(count < 12):
			count += 1
			new_t_list.append(t)
			new_d_list.append(sum(sum_d) / count)
			new_v_list.append(sum(sum_v) / count)
			
		elif(count == 12):
			sum_d = sum_d[(-1 * count):]
			sum_v = sum_v[(-1 * count):]
			new_t_list.append(t)
			new_d_list.append(sum(sum_d) / count)
			new_v_list.append(sum(sum_v) / count)
			
		else:
			print("count is larger than 12")
			sys.exit()

	return new_t_list, new_d_list, new_v_list
	
#差分フィルタ (1階微分)
def differ_filter(t_list, d_list, v_list):
	new_t_list = []
	new_d_list = []
	new_v_list = []
	d_before = d_list[:-1]
	d_before.insert(0, 0)
	d_after = d_list[1:]
	d_after.append(0)
	d_mat = np.array([d_before, d_list, d_after])	 #3列の行列を作成[[f-], [f], [f+]]
	v_before = v_list[:-1]
	v_before.insert(0, 0)
	v_after = v_list[1:]
	v_after.append(0)
	v_mat = np.array([v_before, v_list, v_after])	 #3列の行列を作成[[f-], [f], [f+]]
	kernel = np.array([[-1, 0, 1]])
	length = len(t_list)
	for i in range(length):
		if(i == 0):
			new_t_list.append(t_list[i])
			new_d_list.append(np.dot(np.array([0, 1 / 2, 1 / 2]), d_mat[:, i]))
			new_v_list.append(np.dot(np.array([0, 1 / 2, 1 / 2]), v_mat[:, i]))
		elif(i == length - 1):
			new_t_list.append(t_list[i])
			new_d_list.append(np.dot(np.array([1 / 2, 1 / 2, 0]), d_mat[:, i]))
			new_v_list.append(np.dot(np.array([1 / 2, 1 / 2, 0]), v_mat[:, i]))			
		else:
			new_t_list.append(t_list[i])
			new_d_list.append(np.dot(kernel, d_mat[:, i]))
			new_v_list.append(np.dot(kernel, v_mat[:, i]))	
	return new_t_list, new_d_list, new_v_list

#ラプラシアン (２階微分)
def differ_filter2(t_list, d_list, v_list):
	new_t_list = []
	new_d_list = []
	new_v_list = []
	d_before = d_list[:-1]
	d_before.insert(0, 0)
	d_after = d_list[1:]
	d_after.append(0)
	d_mat = np.array([d_before, d_list, d_after])	 #3列の行列を作成[[f-], [f], [f+]]
	v_before = v_list[:-1]
	v_before.insert(0, 0)
	v_after = v_list[1:]
	v_after.append(0)
	v_mat = np.array([v_before, v_list, v_after])	 #3列の行列を作成[[f-], [f], [f+]]
	kernel = np.array([[1, -2, 1]])
	length = len(t_list)
	for i in range(length):
		if(i == 0):
			new_t_list.append(t_list[i])
			new_d_list.append(np.dot(np.array([0, -1, 1]), d_mat[:, i]))
			new_v_list.append(np.dot(np.array([0, -1, 1]), v_mat[:, i]))
		elif(i == length - 1):
			new_t_list.append(t_list[i])
			new_d_list.append(np.dot(np.array([-1, 1, 0]), d_mat[:, i]))
			new_v_list.append(np.dot(np.array([-1, 1, 0]), v_mat[:, i]))			
		else:
			new_t_list.append(t_list[i])
			new_d_list.append(np.dot(kernel, d_mat[:, i]))
			new_v_list.append(np.dot(kernel, v_mat[:, i]))	
	return new_t_list, new_d_list, new_v_list

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
	time_list, distance_list, velocity_list = read_gps() #2次元リスト (1日分 * 日数分)
	for (t_list, d_list, v_list) in zip(time_list, distance_list, velocity_list):
		print(len(t_list))
		t_list, d_list, v_list = median_for_minutes(t_list, d_list, v_list)
		print(len(t_list))
		t_list, d_list, v_list = convo_per_minutes(t_list, d_list, v_list)
		#print(len(t_list))
		#t_list, d_list, v_list = mean_per_minutes(t_list, d_list, v_list)
		print(len(t_list))
		t_list, d_list, v_list =  differ_filter(t_list, d_list, v_list)
		#print(len(t_list))
		#t_list, d_list, v_list =  differ_filter2(t_list, d_list, v_list)
		print(len(t_list))
		plot_velocity_data(t_list, d_list, v_list)
	#df1 = pd.DataFrame(velocity_list)