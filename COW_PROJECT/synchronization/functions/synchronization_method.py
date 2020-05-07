#-*- encoding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import math
import csv
import datetime
import sys
import os

# 自作メソッド
#import dtw


def make_vector(behavior_list):
    """ 行動リストからベクトルを生成 """
    vector = np.zeros(3)
    for behavior in behavior_list:
        if (behavior == 0): # 休息
            vector[0] += 1
        elif (behavior == 1): # 採食
            vector[1] += 1
        else: # 歩行
            vector[2] += 1
    return vector


def measure_synchronization(vector1, vector2):
    """ 2頭間の行動同期を定量化する """
    vector1_magnitude = np.linalg.norm(vector1, ord=2) # ベクトル１の大きさ
    vector2_magnitude = np.linalg.norm(vector2, ord=2) # ベクトル２の大きさ
    inner_product = np.dot(vector1, vector2) # 内積
    print(vector1_magnitude, vector2_magnitude, inner_product)
    return inner_product / (vector1_magnitude * vector2_magnitude) # コサイン類似度


def average_behavior(community_member, community_data):
    """ コミュニティメンバのコサイン類似度の平均を求める """
    vector_list = []
    for cow_id, behavior_list in zip(community_member, community_data):
        vector = make_vector(behavior_list)
        vector_list.append(vector)
    count = 1
    similarity = 0.0
    for cow_id1, behavior_list1, vector1 in zip(community_member, community_data, vector_list):
        for cow_id2, behavior_list2, vector2 in zip(community_member, community_data, vector_list):
            if (cow_id1 != cow_id2):
                similarity += measure_synchronization(vector1, vector2)
                count += 1
    return similarity / count


