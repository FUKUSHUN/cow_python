#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import StratifiedKFold # 交差検証用
from sklearn.model_selection import cross_val_score # 交差検証用
from sklearn import tree # 可視化用
import pydotplus as pdp # 可視化用


filename = "training_data.csv"
data_set = pd.read_csv(filename, sep = ",", header = None, usecols = [2,3,5,6,8,14,15], names=('RTime', 'WTime', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2'))
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
filename1 = 'rf/model.pickle'
pickle.dump(model1, open(filename1, 'wb'))
filename2 = 'rf/model2.pickle'
pickle.dump(model2, open(filename2, 'wb'))

# 生成された木の1個目を可視化
estimator1 = model1.estimators_[0]
estimator2 = model1.estimators_[0]
filename1 = "rf/tree1_1.png"
filename2 = "rf/tree2_1.png"
dot_data1 = tree.export_graphviz(
            estimator1,
            out_file=None,
            filled=True,
            rounded=True,
            feature_names=np.array(['RTime', 'WTime', 'Velocity', 'MVelocity', 'Distance']),
            class_names="Target1",
            special_characters=True
            )
dot_data2 = tree.export_graphviz(
            estimator2,
            out_file=None,
            filled=True,
            rounded=True,
            feature_names=np.array(['RTime', 'WTime', 'Velocity', 'MVelocity', 'Distance']),
            class_names="Target2",
            special_characters=True
            )
graph = pdp.graph_from_dot_data(dot_data1)
graph.write_png(filename1)
graph = pdp.graph_from_dot_data(dot_data2)
graph.write_png(filename2)
