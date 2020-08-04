import os, sys
import time
import datetime
import statistics
import numpy as np
import pandas as pd
import pdb

class Preprocessor:
    cow_id_list:list

    def __init__(self, cow_id_list):
        self.cow_id_list = cow_id_list

    def preprocess(self, W_list, communities_list, d):
        """ 時系列のコミュニティとインタラクショングラフのリストから入力データと出力データの系列を作成する
            W_list:     インタラクショングラフの時系列リスト
            communities_list:   コミュニティの時系列リスト
            d:          過去の重みの参照個数（入力データの次元数） """
        standardized_W_list = []
        for W in W_list:
            new_W = self._standardize(W)
            standardized_W_list.append(new_W)
        X, y = self._create_train_data(standardized_W_list, communities_list, d)
        # 入力がすべて0のものは除外する
        removed_indexes = []
        for i, x in enumerate(X):
            if (np.all(x == 0)):
                removed_indexes.append(i)
        X = np.delete(X, obj=removed_indexes, axis=0)
        y = np.delete(y, obj=removed_indexes, axis=None)
        return X, y

    def _create_train_data(self, W_list:list, communities_list, d):
        """ データを加工し教師データを作成する
            W_list: 標準化済みのインタラクショングラフWのリスト（d個）
            commiunities_list: 時刻t (予測するための) コミュニティ """
        X, y = [], []
        input_W_list = []
        for i, W in enumerate(W_list):
            if (i < d):
                input_W_list.append(W)
            else:
                communities = communities_list[i]
                _X, _y = self._make_train_at_t(input_W_list, communities)
                X.extend(_X)
                y.extend(_y)
                input_W_list.pop(0)
                input_W_list.append(W)
        return np.array(X), np.array(y)

    def _make_train_at_t(self, input_W_list, communities):
        """ 時刻tでの入力データと出力データを作成する
            （時刻tで同じコミュニティならば1 そうでなければ0とし，時刻t-1以前のインタラクショングラフから予測） """
        input_rows = []
        output_rows = []
        for i, cow_id1 in enumerate(self.cow_id_list):
            for j, cow_id2 in enumerate(self.cow_id_list):
                if (i < j):
                    inputs = []
                    for W in input_W_list:
                        inputs.append(W[i,j])
                    input_rows.append(inputs)
                    output = self._checkif_same_community(communities, cow_id1, cow_id2)
                    output_rows.append(output)
        return input_rows, output_rows

    def _checkif_same_community(self, communities, cow_id1, cow_id2):
        """ cow_id1, cow_id2が同じコミュニティかどうかチェックする """
        for com in communities:
            if (cow_id1 in com and cow_id2 in com):
                return 1
            elif (cow_id1 in com or cow_id2 in com):
                return 0
        return 0

    def _standardize(self, W:np.array):
        """ インタラクショングラフを標準化する """
        K = len(W)
        elements = []
        for i in range(K):
            for j in range(K):
                elements.append(W[i,j])
        mean = statistics.mean(elements)
        pstdev = statistics.pstdev(elements)
        for i in range(K):
            for j in range(K):
                if (W[i,j] != 0):
                    W[i,j] = (W[i,j] - mean) / pstdev + 5 # 0, 0, ..., 0を見分けるため、平均5、標準偏差1に標準化する
        return W