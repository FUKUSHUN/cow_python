import numpy as np
import pandas as pd
import pdb

class DBSCAN:
    eps: float
    minPts: int

    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts

    def fit(self, X):
        """ クラスタリングを行うメインメソッド
        Parameter
        X:np.array (K*K行列)    各ノード間の距離を行列として保持しておく（scikit-learnのXとは異なるので注意）
        Return
        cluster: np.array (K行列)   クラスタリング結果. 0はノイズ1以降がクラスタとしてクラスタ番号を格納 """
        K = len(X) # データ数
        cluster = np.zeros(K) # クラスタ結果
        cluster_num = 1 # クラスタ番号
        core_list = [] # コア点のリスト
        visited_points = [] # コア点の中ですでに訪問済みの点を格納
        # --- コア点を探索する. その点以外は一旦ノイズ（非クラスタとする）---
        neighbor_list = self._make_neighbors_list(X)
        for i in range(K):
            if (self._is_core(neighbor_list, i)):
                core_list.append(i)
        # --- 未訪問のコア点からクラスタを拡大していく ---
        for c in core_list:
            reachable_points = self._expand_cluster(neighbor_list, c)
            if (not c in visited_points):
                for p in reachable_points:
                    cluster[p] = cluster_num # 自分自身も含んでいる
                    visited_points.append(p)
                cluster_num += 1
            else:
                continue
        return cluster

    def _make_neighbors_list(self, X):
        """ 各点の近傍点リストを作成する
            Return
            neighbor_list: list (2D)    各点の近傍点のリスト """
        neighbor_list = []
        for i in range(len(X)):
            neighbors = []
            for j in range(len(X)):
                if (i != j and X[i,j] <= self.eps):
                    neighbors.append(j)
            neighbor_list.append(neighbors)
        return neighbor_list

    def _is_core(self, neighbor_list, num):
        """ コア点かどうかを調査する
            Return
            True or False:  コア点ならTrue """
        if (len(neighbor_list[num]) >= self.minPts):
            return True
        else:
            return False

    def _expand_cluster(self, neighbor_list, index):
        """ コア点から到達可能な点を探索していく
            Return
            reachable_points:   指定したコア点を"含む"クラスタ集合 """
        reachable_points = [index] # indexコアの到達可能点を格納
        reachable_points = self.__expand(neighbor_list, reachable_points, index)
        return reachable_points

    def __expand(self, neighbor_list, reachable_points, pts):
        """ 再帰的にクラスタを拡張する """
        for p in neighbor_list[pts]:
            if (not p in reachable_points):
                reachable_points.append(p)
                reachable_points = self.__expand(neighbor_list, reachable_points, p)
        return reachable_points