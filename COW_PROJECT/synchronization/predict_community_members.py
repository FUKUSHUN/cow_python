import os, sys
import time
import datetime
import numpy as np
import pandas as pd
import pdb
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import pickle

#自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.community_analyzer as commnity_analyzer
import synchronization.interaction_analyzer as interaction_analyzer
import synchronization.functions.utility as my_utility
import synchronization.prediction_model.preprocess as prediction
from synchronization.prediction_model.neural_network import LearnerForNeuralNetwork
from synchronization.graph_operation.graph_series import GraphSeriesAnalysis

from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score) # for evaluation
def test(clf, test_X:np.array, test_y:np.array):
    """ テストを行う
        X: np.array(2D)
        y: np.array(1D) """
    pred_y, proba_y = predict(clf, test_X)
    accuracy, precision, recall, f_measure, auc = accuracy_score(test_y, pred_y), precision_score(test_y, pred_y), \
        recall_score(test_y, pred_y), f1_score(test_y, pred_y), roc_auc_score(test_y, proba_y[:,1])
    return accuracy, precision, recall, f_measure, auc

def predict(model, X:np.array):
    """ 新しい入力に対して予測を行い，予測と予測確率を出力する
        X: np.array(2D)
        model: sklearn.base.ClassifierMixin """
    y = model.predict(X)
    y_proba = model.predict_proba(X)
    return y, y_proba

if __name__ == "__main__":
    delta_c = 2 # コミュニティの抽出間隔 [minutes]
    delta_s = 5 # データのスライス間隔 [seconds] 
    epsilon = 12 # コミュニティ決定のパラメータ
    dzeta = 12 # コミュニティ決定のパラメータ
    leng = 5 # コミュニティ決定のパラメータ
    start = datetime.datetime(2018, 10, 21, 0, 0, 0)
    end = datetime.datetime(2018, 10, 24, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    change_point_file = "./synchronization/change_point/"
    output_file = "./synchronization/output/"
    date = start
    target_list = ['20113','20170','20295','20299']
    while (date < end):
        s1 = time.time()
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        analyzer = commnity_analyzer.CommunityAnalyzer(cow_id_list) # 牛のリストに更新があるため、必ずSynchronizerの後にする
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="behavior", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, visualized_g=False, visualized_m=False, delta=delta_s, leng=leng) \
                if (t_start <= t) else [[]] # コミュニティを決定
            analyzer.append_community([t, community])
            analyzer.append_graph([t, interaction_graph])
            t += datetime.timedelta(minutes=delta_c)
        e1 = time.time()
        print("処理時間", (e1-s1)/60, "[min]")
        # --- 次のコミュニティを予測する ---
        s2 = time.time()
        interaction_graph_list = [graph for t, graph in analyzer.graph_list]
        graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list)
        for cow_id in target_list:
            change_points, score_list = graph_analyzer.detect_change_point(cow_id, t_list)
            df = pd.concat([pd.Series(t_list), pd.Series(score_list)], axis=1, names=["time", "score"])
            df.to_csv("test.csv")
            graph_analyzer.visualize_graph(cow_id, t_list)
            pdb.set_trace()
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        pdb.set_trace()
