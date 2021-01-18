import os, sys
import csv
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

"""
3パターンの行動系列から変化点を抽出する
"""

def cut_point_search(series, threshold = 45):
    """ 行動の変化点を検出する
        Return
            change_point_series: list 変化点ならば1, でなければ0 """
    idx = 0
    stack = [(series, idx)] # セグメントの候補をスタックする（データ集合，開始インデックス）
    change_points = [idx]
    while (len (stack) > 0):
        behaviors, idx = stack.pop(0)
        theta_0 = estimate_parameters(behaviors)
        base_score = _measure_log_probability(behaviors, theta_0)
        score_list = [0] # 1番最初の要素の変化点スコアは0とする
        for i in range(1, len(behaviors)):
            theta_1 = estimate_parameters(behaviors[:i])
            theta_2 = estimate_parameters(behaviors[i:])
            likelihood = _measure_log_probability(behaviors[:i], theta_1) + _measure_log_probability(behaviors[i:], theta_2) - base_score
            score_list.append(likelihood)
        change_point_score = max(score_list)
        if (threshold <= change_point_score):
            change_point_idx = score_list.index(max(score_list))
            segment1, segment2 = behaviors[:change_point_idx], behaviors[change_point_idx:]
            stack.extend([(segment1, idx), (segment2, idx + change_point_idx)]) # セグメントを分割してstackに追加
            change_points.append(idx + change_point_idx)
    change_point_series = [1 if i in change_points else 0 for i in range(len(series))] # 変化点のみフラグを立てる
    return change_point_series


def estimate_parameters(series):
    """ 3種類の行動パラメータを推定する
        Return
            theta:  np.array    3次元ベクトルのカテゴリ分布のパラメータ """
    theta = np.zeros(3)
    for b in series:
        if (b == 0):
            theta[0] += 1
        elif (b == 1):
            theta[1] += 1
        else:
            theta[2] += 1
    theta /= sum(theta) # 正規化
    return theta

def _measure_log_probability(series, theta):
    """ 負の対数尤度を計算する """
    log_p = sum([np.log(theta[b]) for b in series])
    return log_p

def _estimate_parameters2(series):
    """ 9種類の行動遷移パラメータを推定する
        Return
            theta: np.array 3×3の遷移行列 """
    theta = np.zeros((3, 3))
    before_b = series[0]
    for i in range(1, len(series)):
        after_b = series[i]
        theta[before_b, after_b] += 1
    theta /= sum(theta)
    return theta