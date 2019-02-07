#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import math
import statistics
import re
import datetime
import gc
import sys
import cows.cow as Cow
import cows.geography as geo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
	
#1分間の総和を求める (5s間隔なので12回分の総和をとるだけ), angleについては角度の平均を求める
def sum_for_minutes(t_list, d_list, a_list):
	count = 0
	new_t_list = []
	new_d_list = []
	new_a_list = []
	sum_d = []
	sum_cos = []
	sum_sin = []
	for (t, d, a) in zip(t_list, d_list, a_list):
		count += 1
		sum_d.append(d)
		if(a != -1):
			sum_cos.append(math.cos(math.radians(a)))
			sum_sin.append(math.sin(math.radians(a)))
		if(count == 12):
			new_t_list.append(datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, 0))
			new_d_list.append(sum_d)
			if(len(sum_cos) != 0):
				c = sum(sum_cos) / len(sum_cos)
				s = sum(sum_sin) / len(sum_sin)
				new_a_list.append(math.degrees(math.atan2(s, c)))
			else:
				new_a_list.append(None)
			count = 0
			sum_d = []
			sum_cos = []
			sum_sin = []
	return new_t_list, new_d_list, new_a_list
	
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