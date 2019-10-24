#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier # 勾配ブースティング
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn import tree # 可視化用
import pydotplus as pdp # 可視化用

# 自作モジュール
import evaluation_of_classifier as evaluation


if __name__ == '__main__':
    usecols = [1,2,3,4,5,6,7,9,12,13]
    names = ('RCategory','WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2')
    filename = os.path.abspath('./') + "/training_data/training_data.csv"
    data_set = pd.read_csv(filename, sep = ",", header = None, usecols = usecols, names=names)
    data_set = data_set.sample(frac=1).reset_index(drop=True) # データをシャッフル
    train_dataset, test_dataset = data_set[:int(0.8 * len(data_set))], data_set[int(0.8 * len(data_set)):] # 訓練データとテストデータに分割
    train_dataset1, test_dataset1 = train_dataset.drop("Target2", axis = 1), test_dataset.drop("Target2", axis = 1) # 停止セグメント
    train_dataset2, test_dataset2 = train_dataset.drop("Target1", axis = 1), test_dataset.drop("Target1", axis = 1) # 活動セグメント

    # モデルのfitとevaluate
    model1 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=777) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
    model1 = evaluation.learn(model1, train_dataset1, "Target1")
    evaluation1 = evaluation.evaluate(model1, test_dataset1, "Target1")
    evaluation.output_csv("./bst/validation_r.csv", model1, test_dataset1, "Target1", ["REST_Proba", "GRAZE_Proba"])

    model2 = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=777) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
    model2 = evaluation.learn(model2, train_dataset2, "Target2")
    evaluation2 = evaluation.evaluate(model2, test_dataset2, "Target2")
    evaluation.output_csv("./bst/validation_a.csv", model2, test_dataset2, "Target2", ["REST_Proba", "GRAZE_Proba", "WALK_Proba"])

    print(evaluation1)
    print(evaluation2)

    # モデルを保存する
    filename1 = 'bst/model.pickle'
    pickle.dump(model1, open(filename1, 'wb'))
    filename2 = 'bst/model2.pickle'
    pickle.dump(model2, open(filename2, 'wb'))

