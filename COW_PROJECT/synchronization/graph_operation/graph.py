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
    graph1: np.array
    graph2: np.array

    def __init__(self, graph1, graph2):
        self.graph1 = graph1
        self.graph2 = graph2
        return

    def measure_similarity(self, focused_index, max_depth=None):
        """ グラフの類似度を測る
            cow_id_list: 牛の個体番号のリスト
            focused_index: 隣接行列の何番目に注目するか（グラフから注目ノードから到達できる部分グラフを探索し比較するために必要）
            element_threshold: 部分グラフのノード数の上限 """
        node_list1, sub_graph1 = self._extract_subgraph(self.graph1, focused_index, max_depth=max_depth)
        node_list2, sub_graph2 = self._extract_subgraph(self.graph2, focused_index, max_depth=max_depth)
        union_node_set = set(node_list1) | set(node_list2)
        cut_graph1 = self._cut_graph(self.graph1, union_node_set)
        cut_graph2 = self._cut_graph(self.graph2, union_node_set)
        # patterns1 = self._extract_patterns(sub_graph1, node_list1, focused_index, max_depth=element_threshold)
        # patterns2 = self._extract_patterns(sub_graph2, node_list2, focused_index, max_depth=element_threshold)
        distance = self._caluculate_similarity(cut_graph1, cut_graph2)
        return distance
    
    def visualize_graph(self, focused_index, label_list, save_path, filename, max_depth=None):
        node_list1, sub_graph1 = self._extract_subgraph(self.graph1, focused_index, max_depth=max_depth)
        cut_graph1 = self._cut_graph(self.graph1, node_list1)
        color_list = [1 if i in node_list1 else 0 for i in range(len(label_list))]
        self._visualize_graph(cut_graph1, label_list, color_list, save_path, filename, weighted=True)
        return

    def _extract_subgraph(self, graph, index, max_depth=None):
        """ 注目ノードを含む部分グラフを抽出する (幅優先探索)
            graph:  np.array 隣接行列で表したグラフ
            index:  int 注目ノードのインデックス
            max_depth:  int 注目ノードからの距離（間に挟むノード1つにつき1）の閾値
            Return
            node_list: list サブグラフのノードのリスト（元のグラフのインデックスを保管）
            sub_graph: np.array サブグラフの隣接行列 """
        node_set = set([index]) # 最終的に到達可能ノードのインデックスを保管する
        unvisited_node_set = set([index]) # 到達可能なノードのうち探索済みでないもの保管する
        visited_node_set = set([]) # 到達可能なノードのうち探索済みのものを保管する
        # --- 到達可能なノードを探索 ---
        depth = 1 # 探索の深さ（注目ノードからどれだけ離れているか）
        while((len(unvisited_node_set) != 0 and max_depth is None) or (max_depth is not None and depth <= max_depth)):
            arrivable_node_set = set([])
            # 幅優先探索
            for i in unvisited_node_set:
                try: 
                    arrivable_node_set = arrivable_node_set | set([j for j, value in enumerate(graph[i]) if value > 0]) # 和集合をとる, あるノードから到達可能なノードをチェックセットに格納する
                except IndexError:
                    pdb.set_trace()            
                visited_node_set.add(i) # 探索済みを追加 (探索済みを更新)
            unvisited_node_set = unvisited_node_set - visited_node_set # 差集合をとる (未探索を更新)
            for j in arrivable_node_set:
                if (not j in visited_node_set): # 探索済みでなければ
                    unvisited_node_set.add(j) # 到達可能ノードを追加 (セットなので重複要素は追加されない, 未探索を更新)
                node_set.add(j) # 到達可能ノードを追加 (セットなので重複要素は追加されない, 到達可能を更新)
            depth += 1 # 深さを更新
        # --- 部分グラフを抽出する ---
        node_list = sorted(list(node_set))
        sub_graph = graph[np.ix_(node_list, node_list)] # 対象の行と列をもとの行列から抜き出す
        return node_list, sub_graph
    
    def _extract_patterns(self, graph, node_list, focused_index, max_depth=None):
        """ グラフからパターン（部分グラフ）を抽出する
            Parameter
            graph: np.array 隣接行列
            node_list: もとのグラフ（全体の隣接行列）における，今のグラフ（注目ノードから到達可能なノードの部分グラフ）の各ノードのインデックス
            focused_index: 注目するノードのインデックス
            max_depth: 注目ノードからの深さの閾値
            Return
            pattern_list:   パターン (チャンク) をリストで表現し返す """
        pattern_list = [] # Return
        last_depth_pattern_list = [] # 最深のノードを含むサブセット
        unvisited_node_set = set([focused_index]) # 到達可能なノードのうち探索済みでないもの保管する
        visited_node_set = set([]) # 到達可能なノードのうち探索済みのものを保管する
        depth = 1 # 部分グラフのレベル（ノード数）
        last_level_conbo = [[i] for i in range(len(graph))] # 前のレベルでのパターン集合
        while ((len(unvisited_node_set) != 0 and max_depth is None) or (max_depth is not None and depth <= max_depth)):
            arrivable_node_set = set([]) # 到達可能ノードのセット
            # 幅優先探索で注目ノードからdepthの深さにあるノードを探索
            for i in unvisited_node_set:
                arrivable_node_set = arrivable_node_set | set([j for j, value in enumerate(graph[i]) if value > 0]) # 和集合をとる, あるノードから到達可能なノードをチェックセットに格納する
                visited_node_set.add(i) # 探索済みを追加 (探索済みを更新)
            unvisited_node_set = unvisited_node_set - visited_node_set # 差集合をとる (未探索を更新)
            # 最深のノードと既存のサブセットを組み合わせて新しいサブセットを探索

            # 最深のノードを追加した結果できたサブセットと最新のノードを組み合わせて新しいサブセットを探索
            depth += 1
        return pattern_list

    def _cut_graph(self, graph, node_list):
        """ グラフから特定のノードのみを切り出す（元のノード関係を維持したままnode_listに含まれない要素にかかる重みを0にする）
            graph:      np.array    元の隣接行列
            node_list:  list    切り出す必要のあるノードのリスト（ノードのインデックスを格納）
            Return
            cut_graph:  np.array    切り出された後のグラフの隣接行列 """
        N = len(graph)
        cut_graph = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if (i in node_list and j in node_list):
                    cut_graph[i, j] = graph[i, j]
        return cut_graph

    def _caluculate_similarity(self, graph1, graph2):
        """ パターン（部分グラフ）からグラフ間の類似度を算出する（算出式については確定ではなく、変更時にはこのメソッド内を変更するとする）
            patterns1, patterns2: list """
        distance = 0.0
        N = len(graph1)
        for i in range(N):
            for j in range(N):
                distance += abs(graph1[i,j] - graph2[i,j])
        return distance / 2 # 向きがないため単純に2倍になる
    
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
        node_colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (0,0,0), (0,0.5,0.5), (0.5,0,0.5),(0.5,0.5,0)]
        color_list = [node_colors[i%len(node_colors)] for i in color_list]
        plt.figure(figsize=(19.2, 9.67))
        # 円状にノードを配置
        pos = {
            n: (np.cos(2*i*np.pi/num_nodes), np.sin(2*i*np.pi/num_nodes))
            for i, n in enumerate(g.nodes)
        }
        #ノードとエッジの描画
        nx.draw_networkx_edges(g, pos, edge_color='y')
        nx.draw_networkx_nodes(g, pos, node_color=color_list, alpha=0.5) # alpha: 透明度の指定
        nx.draw_networkx_labels(g, pos, font_size=10) #ノード名を付加
        if (weighted):
            edge_labels = {(i, j): w['weight'] for i, j, w in g.edges(data=True)}
            nx.draw_networkx_edge_labels(g,pos, edge_labels=edge_labels)
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
    graph3 = np.array([[0, 0.5, 1, 0, 0, 0], [0.5, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0]])
    graph4 = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0]])
    ga = GraphAnalysis(graph3, graph4)
    ga.measure_similarity(0, max_depth=2)
    ga.visualize_graph(0, ["10", "20", "30", "40", "50", "60"], "./test/", "graph.png", max_depth=3)