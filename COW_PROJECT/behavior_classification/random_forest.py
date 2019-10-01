#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import csv
import datetime
import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import StratifiedKFold # 交差検証用
from sklearn.model_selection import cross_val_score # 交差検証用
from sklearn import tree # 可視化用
import pydotplus as pdp # 可視化用

# 自作クラス
import output_features

def get_existing_cow_list(date:datetime):
    """
    引数の日にちに第一放牧場にいた牛のリストを得る
    """
    path = os.path.abspath('./') + "\\behavior_classification\\" + date.strftime("%Y-%m") + ".csv"
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if (datetime.datetime.strptime(row[0], "%Y/%m/%d") == date):
                return row[1:]

    print("指定の日付の牛のリストが見つかりません", date.strftime("%Y/%m/%d"))
    sys.exit()


def make_features_files(date, cow_id_list):
    """
    指定した牛のリストの特徴出力を行う
    """
    for cow_id in cow_id_list:
        filename = os.path.abspath('./') + "\\behavior_classification\\training_data\\" + date.strftime("%Y%m%d") + "_" + str(cow_id) + ".csv"
        output_features.output_features(filename, date, cow_id)


if __name__ == '__main__':
    filename = os.path.abspath('./') + "\\training_data\\training_data.csv"
    data_set = pd.read_csv(filename, sep = ",", header = None, usecols = [1,2,3,4,5,6,7,9,12,13], names=('RCategory','WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2'))
    train_data_set1 = data_set.drop("Target2", axis = 1)
    train_data_set2 = data_set.drop("Target1", axis = 1)

    x = pd.DataFrame(train_data_set1.drop("Target1", axis = 1))
    y = pd.DataFrame(train_data_set1["Target1"])

    w = pd.DataFrame(train_data_set2.drop("Target2", axis = 1))
    z = pd.DataFrame(train_data_set2["Target2"])

    # 説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.15)
    W_train, W_test, Z_train, Z_test = train_test_split(w,z,test_size=0.15)
    X_train, X_test, Y_train, Y_test = X_train.values, X_test.values, np.ravel(Y_train.values), np.ravel(Y_test.values)
    W_train, W_test, Z_train, Z_test = W_train.values, W_test.values, np.ravel(Z_train.values), np.ravel(Z_test.values)

    model1 = RandomForestClassifier(random_state=0, max_depth=5, n_estimators=10) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
    model1 = model1.fit(X_train, Y_train)
    prediction = model1.predict(X_test)
    evaluation1 = accuracy_score(prediction, Y_test)

    model2 = RandomForestClassifier(random_state=0, max_depth=5, n_estimators=10) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
    model2 = model2.fit(W_train, Z_train)
    prediction = model2.predict(W_test)
    evaluation2 = accuracy_score(prediction, Z_test)

    print(evaluation1)
    print(evaluation2)

    # モデルを保存する
    #filename1 = 'rf/model.pickle'
    #pickle.dump(model1, open(filename1, 'wb'))
    #filename2 = 'rf/model2.pickle'
    #pickle.dump(model2, open(filename2, 'wb'))

    # --- 半教師あり学習 ---
    #os.chdir('../') # チェンジディレクトリ
    #date = datetime.datetime(2018, 12, 20, 0, 0, 0)
    #cow_id_list = get_existing_cow_list(date)
    #make_features_files(date, cow_id_list)

