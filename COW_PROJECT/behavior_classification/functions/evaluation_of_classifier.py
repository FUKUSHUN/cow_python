#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import csv
import datetime
import sys
import os
import pickle
from sklearn.externals import joblib # 保存したモデルの復元
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import StratifiedKFold # 交差検証用
from sklearn.model_selection import cross_val_score # 交差検証用


"""
このファイルではscikit-learnの分類器に対してfit, evaluateを行うプログラムをまとめています．
また，詳しい分析のためにテスト結果のCSV出力を行います．突っ込んだ分析はエクセルのマクロ側で処理することを想定しています
"""
def learn(model, train_dataset, target_labelname):
    """ データから学習を行う一連の処理を実装
    Parameter
        model   : scikit-learnのモデル
        train_dataset   : 訓練用データセット    : pandas.DataFrame
        target_labelname    : ターゲット変数の系列の列名 """
    X = pd.DataFrame(train_dataset.drop(target_labelname, axis = 1))
    y = pd.DataFrame(train_dataset[target_labelname])
    X, y = X.values, np.ravel(y.values)
    model = model.fit(X, y)
    return model


def evaluate(model, test_dataset, target_labelname):
    """ データから評価を行う一連の処理を実装
    Parameter
        model   : scikit-learnのモデル
        test_dataset   : テスト用データセット    : pandas.DataFrame
        target_labelname    : ターゲット変数の系列の列名 """
    X = pd.DataFrame(test_dataset.drop(target_labelname, axis = 1))
    y = pd.DataFrame(test_dataset[target_labelname])
    X, y = X.values, np.ravel(y.values)
    pred = model.predict(X)
    evaluation = accuracy_score(pred, y)
    return evaluation


def output_csv(filepath, model, test_dataset, target_labelname, classified_targetname):
    """ CSVファイルを出力する
    Parameter
        filepath    : String型．出力するファイルの保存場所とファイル名
        model   : scikit-learnのモデル
        test_dataset   : テスト用データセット    : pandas.DataFrame
        target_labelname    : ターゲット変数の系列の列名 """
    print(sys._getframe().f_code.co_name, "実行中")
    print("分類器のテスト結果をCSV出力します---")

    X = pd.DataFrame(test_dataset.drop(target_labelname, axis = 1))
    y = pd.DataFrame(test_dataset[target_labelname])
    X, y = X.values, np.ravel(y.values)
    pred = model.predict(X)
    pred_prob = model.predict_proba(X)
    pred_prob_max = np.array([ps.max() for ps in pred_prob])

    # データフレーム化
    test_dataset = test_dataset.reset_index()
    pred = pd.DataFrame(pred, columns=["Prediction"]).reset_index()
    pred_prob = pd.DataFrame(pred_prob, columns=classified_targetname).reset_index()
    pred_prob_max = pd.DataFrame(pred_prob_max, columns=["Probability"]).reset_index()
    df = pd.concat([test_dataset, pred, pred_prob, pred_prob_max], axis=1)
    df = df.drop("index", axis = 1)
    
    # データの書き込み
    df.to_csv(filepath)
    
    print("---"+ filepath + "への出力が終了しました")
    print(sys._getframe().f_code.co_name, "正常終了\n")


def output_roc(filepath, model):
    """ ROC曲線を出力する
    Parameter
        model   : 検証対象のモデル (scikit-learn) """
    print(sys._getframe().f_code.co_name, "実行中")
    print("分類器のROC曲線を出力します---")

    # to do

    print("---" + filepath + "への出力が終了しました")
    print(sys._getframe().f_code.co_name, "正常終了\n")


def calculate_auc(model):
    """ AUCを計算する
    Parameter
        model   : 検証対象のモデル (scikit-learn) """
    print("hogehoge")