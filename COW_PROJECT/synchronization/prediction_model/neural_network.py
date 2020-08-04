import datetime
import numpy as np
from statistics import mean # to culculate mean
from sklearn.model_selection import cross_val_score, cross_validate # for cross validation
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score) # for evaluation
from sklearn.neural_network import MLPClassifier # for Neural Network
# from sklearn.tree import DecisionTreeClassifier # for Decision Tree
# from sklearn.ensemble import RandomForestClassifier # for Random Forest
from sklearn.svm import SVC # for SVM
# from dtreeviz.trees import dtreeviz # for visualization of decision tree
from sklearn.model_selection import GridSearchCV # for grid search
from sklearn.base import BaseEstimator

class Learner:
    clf: BaseEstimator
    candidate_params: dict

    def __init__(self, clf, candidate_params, fold, scoring):
        self.candidate_params = candidate_params
        self.clf = GridSearchCV(estimator=clf, 
                                param_grid = self.candidate_params,   
                                scoring=scoring,       #metrics
                                cv = fold,             #cross-validation
                                n_jobs = 1)            #number of core

    def fit(self, train_X, train_y):
        """ 訓練しベストパラメータを算出する
            X: np.array(2D)
            y: np.array(1D) """
        self.clf.fit(train_X, train_y)
        best_params = self.clf.best_estimator_
        return best_params
    
    def test(self, test_X:np.array, test_y:np.array):
        """ テストを行う
            X: np.array(2D)
            y: np.array(1D) """
        pred_y, proba_y = self.predict(self.clf, test_X)
        accuracy, precision, recall, f_measure, auc = accuracy_score(test_y, pred_y), precision_score(test_y, pred_y), \
            recall_score(test_y, pred_y), f1_score(test_y, pred_y), roc_auc_score(test_y, proba_y[:,1])
        return accuracy, precision, recall, f_measure, auc

    def predict(self, model, X:np.array):
        """ 新しい入力に対して予測を行い，予測と予測確率を出力する
            X: np.array(2D)
            model: sklearn.base.ClassifierMixin """
        y = model.predict(X)
        y_proba = model.predict_proba(X)
        return y, y_proba

class LearnerForNeuralNetwork(Learner):

    def __init__(self, candidate_params, fold, scoring):
        clf = MLPClassifier(random_state=1)
        super().__init__(clf, candidate_params, fold, scoring)

    def fit(self, train_X, train_y):
        best_params = super().fit(train_X, train_y)
        return best_params
    
    def test(self, test_X, test_y):
        accuracy, precision, recall, f_measure, auc = super().test(test_X, test_y)
        return accuracy, precision, recall, f_measure, auc
