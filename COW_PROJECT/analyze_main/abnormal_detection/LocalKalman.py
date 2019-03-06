#-*- encoding:utf-8 -*-
import numpy as np
import scipy.optimize as sp
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class KalmanFilter:
	sig_v2:float
	sig_w2:float
	train_data_list = []
	
	def __init__(self, v, w, train_data_list):
		self.train_data_list = train_data_list
		self.estimateParameter(v, w)
		
	def estimateParameter(self, v, w):
		x0 = np.array([math.log(v), math.log(w)])
		likelihood = sp.minimize(self.getLogLikelihood, x0, method = 'L-BFGS-B')
		x, y = likelihood.x
		self.sig_v2 = math.exp(x)
		self.sig_w2 = math.exp(y)
		print(likelihood)
		
	def getLogLikelihood(self, params):
		v, w = params
		self.sig_v2 = math.exp(v)
		self.sig_w2 = math.exp(w)
		error_list, var_list = self.getErrorSeries()
		sum = 0.0
		for i in range(len(error_list)):
			if(var_list[i] != 0):
				sum += math.log(var_list[i]) + error_list[i] ** 2 / var_list[i]
			else:
				continue
		return sum / 2
		
	def getErrorSeries(self):
		error_list = [] #誤差の系列
		var_list = [] #誤差の分散の系列O
		
		x_t = 1.0 #状態推定値の初期値
		p_t = 1.0 #誤差分散の初期値
		for y in self.train_data_list:
			x_before, p_before = self.predictionStep(x_t, p_t) #予測ステップ
			g_t, x_t, p_t = self.filteringStep(x_before, p_before, y) #フィルタリングステップ
			
			#結果の格納
			error_list.append(y - x_before)
			var_list.append(p_t)
		return error_list, var_list
		
	def sequentialProcess(self, data_list):
		ab_scores = [] #異常度
		x_scores = [] #状態推定値
		p_scores = [] #事後誤差分散
		k_scores = [] #カルマンゲイン
		
		x_t = 0 #状態推定値の初期値
		p_t = 0 #誤差分散の初期値
		for y in data_list:
			a_t = self.abScore(x_t, y, p_t) #異常度の算出
			x_before, p_before = self.predictionStep(x_t, p_t) #予測ステップ
			g_t, x_t, p_t = self.filteringStep(x_before, p_before, y) #フィルタリングステップ
			
			#結果の格納
			ab_scores.append(a_t)
			x_scores.append(x_t)
			p_scores.append(p_t)
			k_scores.append(g_t)
		return ab_scores, x_scores, p_scores, k_scores
			
	"""
        Parameters
        ----------
        p_former		:float	:1時刻前の事後誤差分散
        x_former    	: float	:1時刻前の状態推定値
	"""			
	def predictionStep(self, x_former, p_former):
		x_before = x_former
		p_before = p_former + self.sig_v2
		return x_before, p_before
	
	"""
        Parameters
        ----------
        p_before		:float	:事前誤差分散
        x_before    	: float	:事前状態推定値
        y		:float 観測値
	"""	
	def filteringStep(self, x_before, p_before, y):
		try:
			g = p_before / (p_before + self.sig_w2) #カルマンゲイン
		except ZeroDivisionError:
			g = 0
		x = x_before + g * (y - x_before) #状態推定値
		p = (1 - g) * p_before #事後誤差分散
		return g, x, p
	
	"""
        Parameters
        ----------
        p_former		:float	:1時刻前の事後誤差分散
        x_former    	: float	:1時刻前の状態推定値
        y		:float 観測値
	"""			
	def abScore(self, x_former, y, p_former):
		a = (y - x_former) ** 2 / (self.sig_w2 + p_former) #異常度
		return a
		
	def get_sig_score(self):
		return self.sig_v2, self.sig_w2
		