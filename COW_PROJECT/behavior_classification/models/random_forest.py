#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn import tree # 可視化用
import pydotplus as pdp # 可視化用

# 自作モジュール
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import functions.evaluation_of_classifier as evaluation


if __name__ == '__main__':
    usecols = [1,2,3,4,5,6,7,9,12,13]
    names = ('RCategory','WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2')
    filename = "./training_data/training_data.csv"
    data_set = pd.read_csv(filename, sep = ",", header = None, usecols = usecols, names=names)
    data_set = data_set.sample(frac=1).reset_index(drop=True) # データをシャッフル
    train_dataset, test_dataset = data_set[:int(0.8 * len(data_set))], data_set[int(0.8 * len(data_set)):] # 訓練データとテストデータに分割
    train_dataset1, test_dataset1 = train_dataset.drop("Target2", axis = 1), test_dataset.drop("Target2", axis = 1) # 停止セグメント
    train_dataset2, test_dataset2 = train_dataset.drop("Target1", axis = 1), test_dataset.drop("Target1", axis = 1) # 活動セグメント

    # モデルのfitとevaluate
    model1 = RandomForestClassifier(random_state=777, max_depth=3, n_estimators=20) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
    model1 = evaluation.learn(model1, train_dataset1, "Target1")
    evaluation1 = evaluation.evaluate(model1, test_dataset1, "Target1")
    evaluation.output_csv("./models/rf/validation_r.csv", model1, test_dataset1, "Target1", ["REST_Proba", "GRAZE_Proba"])

    model2 = RandomForestClassifier(random_state=777, max_depth=3, n_estimators=20) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
    model2 = evaluation.learn(model2, train_dataset2, "Target2")
    evaluation2 = evaluation.evaluate(model2, test_dataset2, "Target2")
    evaluation.output_csv("./models/rf/validation_a.csv", model2, test_dataset2, "Target2", ["REST_Proba", "GRAZE_Proba", "WALK_Proba"])

    print(evaluation1)
    print(evaluation2)

    # モデルを保存する
    filename1 = 'models/rf/model.pickle'
    pickle.dump(model1, open(filename1, 'wb'))
    filename2 = 'models/rf/model2.pickle'
    pickle.dump(model2, open(filename2, 'wb'))

    """
    # 生成された木の1個目を可視化
    for i in range(len(model1.estimators_)):
        estimator1 = model1.estimators_[i]
        estimator2 = model2.estimators_[i]
        filename1 = "rf/tree1_" + str(i) + ".png"
        filename2 = "rf/tree2_" + str(i) + ".png"
        dot_data1 = tree.export_graphviz(
                    estimator1,
                    out_file=None,
                    filled=True,
                    rounded=True,
                    feature_names=np.array(names[:-2]),
                    class_names="Target1",
                    special_characters=True
                    )
        dot_data2 = tree.export_graphviz(
                    estimator2,
                    out_file=None,
                    filled=True,
                    rounded=True,
                    feature_names=np.array(names[:-2]),
                    class_names="Target2",
                    special_characters=True
                    )
        graph = pdp.graph_from_dot_data(dot_data1)
        graph.write_png(filename1)
        graph = pdp.graph_from_dot_data(dot_data2)
        graph.write_png(filename2)
        """