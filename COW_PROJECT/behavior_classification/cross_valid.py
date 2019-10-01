#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # 決定木
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.svm import SVC # サポートベクターマシーン
from sklearn.multiclass import OneVsRestClassifier # サポートベクターマシーン
from sklearn.ensemble import GradientBoostingClassifier # 勾配ブースティング
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import StratifiedKFold # 交差検証用 (各クラスが等しくなるように分割)
from sklearn.model_selection import ShuffleSplit # 交差検証用 (完全にシャッフルして分割)
from sklearn.model_selection import cross_val_score # 交差検証用


filename = "training_data/training_data.csv"
data_set = pd.read_csv(filename, sep = ",", header = None, usecols = [1,2,3,4,5,6,7,9,12,13], names=('RCategory','WCategory', 'RTime', 'WTime', 'AccumratedDis', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2'))
train_data_set1 = data_set.drop("Target2", axis = 1)
train_data_set2 = data_set.drop("Target1", axis = 1)

x = pd.DataFrame(train_data_set1.drop("Target1", axis = 1))
y = pd.DataFrame(train_data_set1["Target1"])
X, Y = x.values, np.ravel(y.values)

# 交差検証1
tr1 = DecisionTreeClassifier(random_state=0)
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
tr1_score = cross_val_score(tr1, X, Y, cv=fold)
print("決定木: ", tr1_score)
print(mean(tr1_score))

rf1 = RandomForestClassifier(random_state=0, max_depth=5, n_estimators=10) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
rf1_score = cross_val_score(rf1, X, Y, cv=fold)
print("ランダムフォレスト: ", rf1_score)
print(mean(rf1_score))

svm1 = SVC(C=1.0, kernel='rbf', gamma=0.01, random_state=0) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
svm1 = OneVsRestClassifier(svm1)
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
svm1_score = cross_val_score(svm1, X, Y, cv=fold)
print("SVM: ", svm1_score)
print(mean(svm1_score))

bst1 = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=3, random_state=0) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
bst1_score = cross_val_score(bst1, X, Y, cv=fold)
print("勾配ブースティング: ", bst1_score)
print(mean(bst1_score))

w = pd.DataFrame(train_data_set2.drop("Target2", axis = 1))
z = pd.DataFrame(train_data_set2["Target2"])
W, Z = w.values, np.ravel(z.values)

# 交差検証2
tr2 = DecisionTreeClassifier(random_state=0)
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
tr2_score = cross_val_score(tr2, W, Z, cv=fold)
print("決定木: ", tr2_score)
print(mean(tr2_score))

rf2 = RandomForestClassifier(random_state=0, max_depth=5, n_estimators=10) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
rf2_score = cross_val_score(rf2, W, Z, cv=fold)
print("ランダムフォレスト: ", rf2_score)
print(mean(rf2_score))

svm2 = SVC(C=1.0, kernel='rbf', gamma=0.01, random_state=0) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
svm2 = OneVsRestClassifier(svm2)
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
svm2_score = cross_val_score(svm2, W, Z, cv=fold)
print("SVM: ", svm2_score)
print(mean(svm2_score))

bst2 = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=3, random_state=0) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
#fold = StratifiedKFold(n_splits=5)
fold = ShuffleSplit(n_splits=5, random_state=0)
bst2_score = cross_val_score(bst2, W, Z, cv=fold)
print("勾配ブースティング: ", bst2_score)
print(mean(bst2_score))