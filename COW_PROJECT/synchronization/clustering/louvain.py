import math
import numpy as np
import pandas as pd
import networkx as nx # ネットワークグラフ
import community # Louvain法
import pdb

class CommunityLouvain:
    louvain = None

    def __init__(self):
        self.louvain = Louvain()
    
    def exchange_undirected_graph(self, W, threshold):
        """ 重みなし無向グラフを作成する """
        K = len(W)
        X = np.zeros([K, K])
        for i in range(K):
            for j in range(i,K):
                if (j <= i):
                    X[i,j] = 1 if threshold <= W[i,j] else 0
                    X[j,i] = 1 if threshold <= W[j,i] else 0
        return X

    def create_community(self, cow_id_list, X):
        """ Louvain法を用いたコミュニティを作る """
        cluster = self.louvain.fit(X)
        communities = self._make_nodes(cow_id_list, cluster)
        return communities

    def _make_nodes(self, cow_id_list, cluster):
        new_cluster = []
        node_list = []
        # クラスタ内の番号をソートする
        for c in cluster:
            new_c = sorted(c)
            new_cluster.append(new_c)
        # 番号を牛の個体番号に置き換える
        for c in new_cluster:
            node_list.append([cow_id_list[i] for i in c])
        return node_list

    def create_graph(self, cow_id_list, X):
        graph = nx.Graph()
        edges = []
        # ノードメンバを登録
        for i, cow_id1 in enumerate(cow_id_list):
            graph.add_node(cow_id1)
            for j, cow_id2 in enumerate(cow_id_list):
                if (X[i,j] == 1):
                    edges.append((cow_id1, cow_id2))
        # エッジを追加
        graph.add_edges_from(edges)
        return graph
    
    def create_weighted_graph(self, cow_id_list, W):
        graph = nx.Graph()
        edges = []
        # ノードメンバを登録
        for i, cow_id1 in enumerate(cow_id_list):
            graph.add_node(cow_id1)
            for j, cow_id2 in enumerate(cow_id_list):
                if (W[i,j] > 0):
                    edges.append((cow_id1, cow_id2, math.floor(W[i,j] * 100) / 100))
        # エッジを追加
        graph.add_weighted_edges_from(edges)
        return graph

class Louvain:
    def fit(self, X:np.array):
        """ Louvain法を用いてコミュニティを抽出する
            Parameter
            X: np.array     隣接行列（2次元行列）
            Return
            cluster: list   コミュニティごとにリスト化したクラスタリング結果（行番号を格納） """
        cluster = []
        # --- すべてのノードを独立したクラスタに格納する ---
        for i in range(len(X)):
            cluster.append([i])
        # --- 行列の総和を求める ---
        M = 0 # 2mと等しい
        for X_i in X:
            for X_i_j in X_i:
                M += X_i_j
        # --- 更新処理を行い，Modularityの極大値を求める ---
        is_end = False
        while (not is_end):
            is_end = True ## 終了フラグを立てておく．更新処理があれば降ろされる
            N = len(X)
            seeds = np.random.permutation(np.arange(N)) # [0 - N-1] までの整数列をシャッフルして格納
            # ランダムなノードに対して
            for i in seeds:
                max_delta_q = 0.0
                k_i = 0
                # Modularityの差分ΔQを最大化するエッジを見つけて
                for j in range(N):
                    if (i != j):
                        delta_q = self._calculate_deltaQ(X, i, j, M)
                        if (delta_q > max_delta_q):
                            max_delta_q = delta_q
                            k_i = j
                # 2つのエッジを統合する
                if (max_delta_q > 0.0):
                    X = self._integrate_X(X, i, k_i)
                    cluster = self._update_cluster(cluster, i, k_i)
                    is_end = False ## 終了フラグを降ろす
                    break ## while文のループに戻る
        return cluster
                
    def _calculate_deltaQ(self, X:np.array, i:int, j:int, M:float):
        """ Modularityの差分ΔQを求める """
        k_i_in = X[i,j]
        k_i = sum(X[i])
        sigma_tot = sum(X[j])
        dq = k_i_in - (k_i * sigma_tot) / M
        return dq

    def _integrate_X(self, X:np.array, i:int, j:int):
        """ 隣接行列のi行i列とj行j列を統合する
            iとjの順番はclusterと対応がとれている必要がある """
        # 行の統合. 挿入→削除の順番でなければならない
        x_ij = X[i,:] + X[j,:]
        X = np.insert(X, i, x_ij, axis=0)
        X = np.delete(X, i+1, axis=0)
        X = np.delete(X, j, axis=0)
        # 列の統合. 挿入→削除の順番でなければならない
        x_ij = X[:,i] + X[:,j]
        X = np.insert(X, i, x_ij, axis=1)
        X = np.delete(X, i+1, axis=1)
        X = np.delete(X, j, axis=1)
        return X

    def _update_cluster(self, cluster:list, i:int, j:int):
        """ クラスタを更新する
            iとjの順番はXと対応がとれている必要がある """
        c_ij = cluster[i] + cluster[j]
        # 挿入と削除 (この順番でなければならない)
        cluster.insert(i, c_ij)
        cluster.pop(i+1) # i+1番目 (元のi番目) を削除
        cluster.pop(j) # j番目を削除
        return cluster