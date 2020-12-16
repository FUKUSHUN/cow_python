import datetime
import sys, os
import math
import numpy as np
import pandas as pd
import networkx as nx # ネットワークグラフ
import matplotlib.pyplot as plt # ネットワークグラフ描画
import pdb

class GraphAnalysis:
    """ 2つのグラフ（隣接行列）を比較し、類似度を判定する機能を持つクラス """
    graphs1: list # グラフ（隣接行列）のリスト
    graphs2: list # グラフ（隣接行列）のリスト
    graph_type: str # エッジの分布の過程

    def __init__(self, graphs1, graphs2, graph_type):
        self.graphs1 = graphs1
        self.graphs2 = graphs2
        self.graph_type = graph_type
        return

    def measure_similarity(self, focused_index):
        """ グラフの非類似度をスコア化する
            focused_index: 隣接行列の何番目に注目するか（グラフから注目ノードから到達できる部分グラフを探索し比較するために必要）"""
        # 隣接行列から注目ノードのベクトルのリストを作成する
        x0_list = self._extract_column(self.graphs1 + self.graphs2, focused_index)
        param0 = self._estimate_params(x0_list)
        x1_list = self._extract_column(self.graphs1, focused_index)
        param1 = self._estimate_params(x1_list)
        x2_list = self._extract_column(self.graphs2, focused_index)
        param2 = self._estimate_params(x2_list)
        dissimilarity = self._caluculate_dissimilarity(x1_list, x2_list, param0, param1, param2)
        return dissimilarity
    
    def _extract_column(self, graphs, index):
        ret = [] 
        for graph in graphs:
            ret.append(graph[:, index]) # 1次元になる
        return ret

    def _estimate_params(self, x_list):
        """ 最尤推定を行い，エッジ確率を求める """
        mu = np.zeros(len(x_list[0])) # 各ノードごとにエッジ確率を求める
        for x in x_list: # リストごとに
            mu += x
        mu /= len(x_list)
        return mu

    def _caluculate_dissimilarity(self, x1_list, x2_list, param0, param1, param2):
        """ パターン（部分グラフ）からグラフ間の非類似度を算出する（算出式については確定ではなく、変更時にはこのメソッド内を変更するとする）
            x1_list, x2_list: list  ベクトルの系列
            param0, param1, param2: np.array    パラメータ（各エッジの生起確率をベクトル化したもの） """
        log_K = 0
        for x in x1_list:
            log_K += self._calculate_log_prob(x, param1) - self._calculate_log_prob(x, param0)
        for x in x2_list:
            log_K += self._calculate_log_prob(x, param2) - self._calculate_log_prob(x, param0)
        return log_K # expにすると数が大きくなりすぎてスコアが分かりにくくなる（論文でもlogのままだった）

    def _calculate_log_prob(self, x, param):
        """ 1時刻分の生起確率の対数を計算する (Bernoulli and Poisson)
            Parameter
            x: np.array ベクトル（隣接行列のi列目を1次元にしたものになる）
            param: np.array ベクトル（xの生起確率）"""
        log_p = 0
        if (self.graph_type == "Bernoulli"):
            for x_j, mu_j in zip(x, param):
                log_p += np.log((mu_j ** x_j) * ((1 - mu_j) ** (1 - x_j))) # ベルヌーイ
        elif (self.graph_type == "Poisson"):
            for x_j, lam_j in zip(x, param):
                try:
                    if (lam_j > 0):
                        log_p += x_j * np.log(lam_j) - self._log_factorial(int(x_j)) + (-1 * lam_j) # ポアソン
                    else:
                        log_p += 1 # パラメータが0の場合，重みがつく確率は0，すなわちx=0となる確率は1
                except:
                    pdb.set_trace()
        return log_p
    
    def _log_factorial(self, x):
        """ log(math.factorial(x))を計算する（オーバーフロー対策） """
        if (x == 0):
            return 1 # 0の階乗は1
        else:
            return sum(map(lambda y: math.log(y), range(1,x+1))) # logをとってから足し合わせる = log(x!)

    def visualize_graph(self, graph, focused_index, label_list, save_path, filename):
        cut_graph1 = self._cut_graph(graph, focused_index)
        color_list = [1 if graph[focused_index, j] > 0 else 0 for j in range(len(label_list))]
        self._visualize_graph(cut_graph1, label_list, color_list, save_path, filename, weighted=True)
        return

    def _cut_graph(self, graph, focused_index):
        """ 注目ノードの列のみを残してグラフを切り出す（他のノードの重みを0に置き換える） """
        ret_graph = np.zeros((len(graph), len(graph.T)))
        for i in range(len(graph)):
            for j in range(len(graph)):
                if (i == focused_index or j == focused_index):
                    ret_graph[i,j] = graph[i,j]
        return ret_graph
    
    def _visualize_graph(self, graph, label_list, color_list:list, save_path:str, filename:str, weighted=False):
        """ インタラクショングラフを描画する
            graph:  np.array
            label_list: list    ラベルに付与するリスト
            color_list: list    ノード数分の長さのリスト．node_colorsのインデックスの数字が入る
            save_path: str      セーブするファイルのパス．この中ではパスが通るかの確認はしない
            filename: str       セーブするファイルの名前
            weighted: bool      重み付きを表示するかどうか """
        g = self._create_nxgraph(label_list, graph)
        num_nodes = len(g.nodes)
        node_colors = [(1,0,0,0.0), (1,0,0,1.0), (1,0,0,0.5), (1,0,0,0.4), (1,0,0,0.3), (1,0,0,0.2), (1,0,0,0.15), (1,0,0,0.1), (1,0,0,0.05),(1,0,0,0.0)]
        color_list = [node_colors[int(i%len(node_colors))] for i in color_list]
        plt.figure(figsize=(19.2, 9.67))
        # 円状にノードを配置
        pos = {
            n: (np.cos(2*i*np.pi/num_nodes), np.sin(2*i*np.pi/num_nodes))
            for i, n in enumerate(g.nodes)
        }
        #ノードとエッジの描画
        nx.draw_networkx_edges(g, pos, edge_color='y')
        nx.draw_networkx_nodes(g, pos, node_color=color_list) # alpha: 透明度の指定
        nx.draw_networkx_labels(g, pos, font_size=10) #ノード名を付加
        if (weighted):
            edge_labels = {(i, j): w['weight'] for i, j, w in g.edges(data=True)}
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        plt.axis('off') #X軸Y軸を表示しない設定
        plt.savefig(save_path + filename)
        plt.close()
        return

    def _create_nxgraph(self, label_list, matrix):
        """ 隣接行列から無向グラフを作成する
            label_list: list    ノードに付与するラベル
            matrix: np.array     隣接行列のグラフ（2次元行列で表現） """
        graph = nx.Graph()
        edges = []
        # ノードメンバを登録
        for i, label1 in enumerate(label_list):
            graph.add_node(label1)
            for j, label2 in enumerate(label_list):
                if (matrix[i,j] > 0):
                    edges.append((label1, label2, math.floor(matrix[i,j] * 100) / 100))
        # エッジを追加
        graph.add_weighted_edges_from(edges)
        return graph

if __name__ == "__main__":
    graph1 = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]])
    graph2 = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
    graph3 = np.array([[0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]])
    graph4 = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    ga = GraphAnalysis(graph3, graph4, "Bernoulli")
    # distance =ga.measure_similarity(graph, 0)
    # print("distance: ", distance)
    ga.visualize_graph(graph3, 0, ["10", "20", "30", "40", "50", "60"], "./", "graph.png")