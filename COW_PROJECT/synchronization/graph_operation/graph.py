import datetime
import sys, os
import numpy as np
import pandas as pd
import pdb

class GraphAnalysis:
    """ 2つのグラフ（隣接行列）を比較し、類似度を判定する機能を持つクラス """
    graph1: np.array
    graph2: np.array

    def __init__(self, graph1, graph2):
        self.graph1 = graph1
        self.graph2 = graph2
        return

    def measure_similarity(self, focused_index, max_depth=None, element_threshold=4):
        """ グラフの類似度を測る
            cow_id_list: 牛の個体番号のリスト
            focused_index: 隣接行列の何番目に注目するか（グラフから注目ノードから到達できる部分グラフを探索し比較するために必要）
            element_threshold: 部分グラフのノード数の上限 """
        node_list1, sub_graph1 = self._extract_subgraph(self.graph1, focused_index, max_depth=max_depth)
        node_list2, sub_graph2 = self._extract_subgraph(self.graph2, focused_index, max_depth=max_depth)
        patterns1 = self._extract_patterns(sub_graph1, node_list1, focused_index, element_threshold=element_threshold)
        patterns2 = self._extract_patterns(sub_graph2, node_list2, focused_index, element_threshold=element_threshold)
        similarity = self._caluculate_similarity(patterns1, patterns2)
        return similarity

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
            arrivable_set = set([])
            for i in unvisited_node_set:
                arrivable_set = arrivable_set | set([j for j, value in enumerate(graph[i]) if value > 0]) # 和集合をとる, あるノードから到達可能なノードをチェックセットに格納する
                visited_node_set.add(i) # 探索済みを追加 (探索済みを更新)
            unvisited_node_set = unvisited_node_set - visited_node_set # 差集合をとる (未探索を更新)
            for j in arrivable_set:
                if (not j in visited_node_set): # 探索済みでなければ
                    unvisited_node_set.add(j) # 到達可能ノードを追加 (セットなので重複要素は追加されない, 未探索を更新)
                node_set.add(j) # 到達可能ノードを追加 (セットなので重複要素は追加されない, 到達可能を更新)
            depth += 1 # 深さを更新
        # --- 部分グラフを抽出する ---
        node_list = sorted(list(node_set))
        sub_graph = graph[np.ix_(node_list, node_list)] # 対象の行と列をもとの行列から抜き出す
        return node_list, sub_graph
    
    def _extract_patterns(self, graph, node_list, focused_index, element_threshold=4):
        """ グラフからパターン（部分グラフ）を抽出する
            Parameter
            graph: np.array 隣接行列
            node_list: もとのグラフ（全体の隣接行列）における，今のグラフ（注目ノードから到達可能なノードの部分グラフ）の各ノードのインデックス
            focused_index: 注目するノードのインデックス
            element_threshold: 要素数の限界（注目ノードからの移動回数を制限したいが時間計算量を考慮し要素数で代替する）
            Return
            pattern_list:   パターン (チャンク) をリストで表現し返す """
        pattern_list = []
        level = 2 # 部分グラフのレベル（ノード数）
        last_level_conbo = [[i] for i in range(len(graph))] # 前のレベルでのパターン集合
        while (len(last_level_conbo) != 0 and level <= element_threshold):
            last_level_conbo_tmp = []
            for conbo in last_level_conbo:
                for i in conbo:
                    for j in range(len(graph)):
                        if (graph[i,j] > 0):
                            tmp = sorted(conbo+[j]) # 前の部分グラフに1つノードを追加して新たな部分グラフの要素を作成
                            tmp_node = [node_list[k] for k in tmp]
                            if (pattern_list.count(tmp_node) < 1 and len(set(tmp)) == level and focused_index in tmp_node): # まだパターンが登録されていない and ノードの重複がない and 注目ノードが部分グラフに含まれる
                                pattern_list.append(tmp_node) # パターンを追加
                                last_level_conbo_tmp.append(tmp) # 一時的に現レベルでのパターンを保管しておく
            last_level_conbo = last_level_conbo_tmp
            level += 1
        return pattern_list

    def _caluculate_similarity(self, patterns1, patterns2):
        """ パターン（部分グラフ）からグラフ間の類似度を算出する（算出式については確定ではなく、変更時にはこのメソッド内を変更するとする）
            patterns1, patterns2: list """
        union_set = [] # setはリストの要素をハッシュ化できないためリストを代用
        product_set = [] # setはリストの要素をハッシュ化できないためリストを代用
        for sub1 in patterns1:
            union_set.append(sub1) # 和集合を更新
            if (sub1 in patterns2):
                product_set.append(sub1) # 積集合を更新
        for sub2 in patterns2:
            if (not sub2 in patterns1):
                union_set.append(sub2) # 和集合を更新
        # 類似度を算出（変更可能性部分）
        denominator = 0.0 # 分母
        for s in union_set:
            denominator += len(s)
        numerator = 0.0 # 分子
        for s in product_set:
            numerator += len(s)
        similarity = 0.0 # 類似度
        if (denominator == 0):
            return similarity
        else:
            similarity = numerator / denominator
        return similarity


if __name__ == "__main__":
    graph1 = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]])
    graph2 = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
    graph3 = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 0, 0]])
    graph4 = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0]])
    ga = GraphAnalysis(graph3, graph4)
    pdb.set_trace()
    ga.measure_similarity(0, max_depth=None)