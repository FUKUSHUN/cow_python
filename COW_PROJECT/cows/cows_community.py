#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import community
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cows_relation as cr #自作クラス
import gps.gps_nmea_data_list as gpslist #自作クラス

"""
    Louvainアルゴリズムを用いてコミュニティを生成する
    df :    pandas.DataFrame    :牛の個体番号とGPSデータリストを所持
    dfの形式はcowshed.Cowshed.get_cow_list()を参照すること
    javaのプログラムをそのまま移植している部分が大きいです
"""
def extract_community(df:pd.DataFrame, threshold):
    #隣接行列を作成する
    matrix = np.zeros((len(df.columns), len(df.columns))) # 行列の作成
    cnt = 0
    ave = 0.0
    for i in range(len(df.columns)):
        for j in range(i, len(df.columns)):
            if(i != j):
                tcr = cr.TwoCowsRelation(df.iloc[1,i], df.iloc[1,j])
                value = tcr.count_near_distance_time(threshold)
                matrix[i,j] = value
                matrix[j,i] = value
                ave += value # sum
                cnt += 1
    ave = ave / cnt #average
    dev = 0.0 #standard deviation
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if(i != j):
                dev += (matrix[i,j] - ave) ** 2
    dev = (dev / cnt) ** (1 / 2)
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if(1 < ((matrix[i, j] - ave) / dev)): #偏差値60以上と同義
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0

    #重みなし無向グラフを作成する
    g = nx.Graph()
    edges = []
    for i in range(len(df.columns)):
        g.add_node(df.iloc[0, i])
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if(matrix[i, j] == 1):
                edges.append((df.iloc[0, i], df.iloc[0, j]))
    g.add_edges_from(edges)
    partition = community.best_partition(g)
    return make_node_list(partition)

#コミュニティごとに各牛の個体番号のリストを作成し，コミュニティのリストを返す 
def make_node_list(partition):
    nodes_list = []
    for com in set(partition.values()):
        nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        nodes_list.append(nodes)
    print(nodes_list)
    return nodes_list