import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # サポートベクターマシーン
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.metrics import (roc_curve, auc, accuracy_score)

filename = "training_data/training_data.csv"
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

model1 = SVC(C=1.0, kernel='rbf', gamma=0.05, random_state=0, probability=True) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
model1 = OneVsRestClassifier(model1)
model1 = model1.fit(X_train, Y_train)
prediction = model1.predict(X_test)
evaluation1 = accuracy_score(prediction, Y_test)

model2 = SVC(C=1.0, kernel='rbf', gamma=0.05, random_state=0, probability=True) # seedの設定。seedを設定しないとモデルが毎回変わるので注意
model2 = OneVsRestClassifier(model2)
model2 = model2.fit(W_train, Z_train)
prediction = model2.predict(W_test)
evaluation2 = accuracy_score(prediction, Z_test)

print(evaluation1)
print(evaluation2)

# モデルを保存する
filename1 = 'svm/model.pickle'
pickle.dump(model1, open(filename1, 'wb'))
filename2 = 'svm/model2.pickle'
pickle.dump(model2, open(filename2, 'wb'))