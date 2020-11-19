import numpy as np
import pandas as pd
import networkx as nx # ネットワークグラフ
import community # Louvain法
import pdb
import matplotlib.pyplot as plt # ネットワークグラフ描画

class Visualizer:

    def visualize_graph(self, X, filename):
        """ インタラクショングラフを描画する """
        save_path = "./"
        g = self._exchange_X(X)
        num_nodes = len(g.nodes)

        plt.figure(figsize=(10, 8))
        # 円状にノードを配置
        pos = {
            n: (np.cos(2*i*np.pi/num_nodes), np.sin(2*i*np.pi/num_nodes))
            for i, n in enumerate(g.nodes)
        }
        #ノードとエッジの描画
        pdb.set_trace()
        nx.draw_networkx_edges(g, pos, edge_color='r')
        nx.draw_networkx_nodes(g, pos, alpha=0.5) # alpha: 透明度の指定
        nx.draw_networkx_labels(g, pos, font_size=10) #ノード名を付加
        edge_labels = {(i, j): w['weight'] for i, j, w in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g,pos, edge_labels=edge_labels)
        plt.axis('off') #X軸Y軸を表示しない設定
        plt.savefig(save_path + filename)
        return

    def _exchange_X(self, W):
        graph = nx.Graph()
        edges = []
        # ノードメンバを登録
        for i in range(len(W)):
            graph.add_node(i)
            for j in range(len(W.T)):
                if (W[i,j] > 0):
                    edges.append((i, j, W[i,j]))
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
        viz = Visualizer()
        viz.visualize_graph(X, "original.png")
        # --- 更新処理を行い，Modularityの極大値を求める ---
        is_end = False
        roop = 1
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
                    viz.visualize_graph(X, str(roop)+".png")
                    roop += 1
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