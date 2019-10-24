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

# 自作クラス
import output_features

def get_existing_cow_list(date:datetime):
    """
    引数の日にちに第一放牧場にいた牛のリストを得る
    """
    path = os.path.abspath('./') + "/behavior_classification/" + date.strftime("%Y-%m") + ".csv"
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if (datetime.datetime.strptime(row[0], "%Y/%m/%d") == date):
                return row[1:]

    print("指定の日付の牛のリストが見つかりません", date.strftime("%Y/%m/%d"))
    sys.exit()


def make_features_data(dir_path, date, cow_id_list, usecols, names):
    """
    指定した牛のリストと日付を元に位置情報を取得後，特徴出力を行い，そのファイルをpandas.DataFrameとして読み込む
    """
    df = pd.DataFrame([])
    for cow_id in cow_id_list:
        filename = dir_path + date.strftime("%Y%m%d") + "_" + str(cow_id) + ".csv" # 保存するファイル名
        is_exist = output_features.output_features(filename, date, cow_id) # 特徴を計算しファイル出力する
        if (is_exist):
            data_set = pd.read_csv(filename, sep = ",", skiprows=[0], usecols=usecols, names=names) # データフレームとして読み込み
            df = pd.concat([df, data_set], axis=0, ignore_index=True) # データを結合する
    return df


def self_semi_supervised_learning(model, train_dataset, test_dataset, candidate_dataset, target_labelname, num):
    """
    半教師あり学習のメインルーチン
    Parameter
        model   : scikit-learnのモデル
        train_dataset   : ラベル付き訓練用データ
        test_dataset   : ラベル付きテスト用データ
        candidate_dataset   : 教師データ候補
        target_labelname    : ターゲット変数の系列の列名
        num : 繰り返し回数
    """
    print("半教師あり学習のルーチンを開始します")
    for i in range(num):
        # 既存のモデルで予測を行い信頼度を算出
        candidate_dataset = predict_probability(model, candidate_dataset) # 予測ラベルと予測信頼度を列に追加
        df = candidate_dataset.sort_values("Probability", ascending=False) # 信頼度の降順にデータセットを並び替える
        # 予測の信頼度を元に訓練データに追加するデータセットを抽出する
        train_dataset = train_dataset.reset_index(drop=True) # インデックスを削除
        df = df.reset_index(drop=True) # インデックスを削除
        #train_df = df[:int(0.01 * len(df))] # 信頼度上位1%のみを抽出
        train_df = df[df["Probability"] >= 0.95]
        # 既存の訓練データと新規の予測データを結合する
        train_df = train_df.rename(columns={"Prediction":target_labelname}) # 予測ラベルを正解ラベルに名称変更
        train_df = train_df.drop("Probability", axis=1) # 確率を消去してラベルあり訓練データに形を合わせる
        train_dataset = pd.concat([train_dataset, train_df], axis=0, ignore_index=True) # 訓練データセットに結合
        # モデルを学習
        model = learn(model, train_dataset, target_labelname)
        score = evaluate(model, test_dataset, target_labelname)
        print(i+1, "回目: ", score)
        # 次の予測データを作成する
        df = df[df["Probability"] < 0.95]
        candidate_dataset = df.drop(["Probability", "Prediction"], axis=1) # 新たに追加した行の削除
        #candidate_dataset = df[int(0.01 * len(df)):] # 信頼度下位99%を抽出
        candidate_dataset = candidate_dataset.reset_index(drop=True) # インデックスを削除
    train_dataset.to_csv("self_train.csv")
    print("半教師あり学習のルーチンが終了しました")
    return model


def co_semi_supervised_learning(model1, model2, train_dataset, test_dataset, candidate_dataset, target_labelname, num):
    """
    半教師あり学習のメインルーチン
    Parameter
        model   : scikit-learnのモデル
        train_dataset   : ラベル付き訓練用データ
        test_dataset   : ラベル付きテスト用データ
        candidate_dataset   : 教師データ候補
        target_labelname    : ターゲット変数の系列の列名
        num : 繰り返し回数
    """
    train_dataset1 = train_dataset
    train_dataset2 = train_dataset
    candidate_dataset1 = candidate_dataset
    candidate_dataset2 = candidate_dataset
    print("半教師あり学習のルーチンを開始します")
    for i in range(num):
        # 既存のモデルで予測を行い信頼度を算出
        candidate_dataset1 = predict_probability(model1, candidate_dataset1) # 予測ラベルと予測信頼度を列に追加
        candidate_dataset2 = predict_probability(model2, candidate_dataset2) # 予測ラベルと予測信頼度を列に追加
        df1 = candidate_dataset1.sort_values("Probability", ascending=False) # 信頼度の降順にデータセットを並び替える
        df2 = candidate_dataset2.sort_values("Probability", ascending=False) # 信頼度の降順にデータセットを並び替える
        # 予測の信頼度を元に訓練データに追加するデータセットを抽出する
        train_dataset1 = train_dataset1.reset_index(drop=True) # インデックスを削除
        train_dataset2 = train_dataset2.reset_index(drop=True) # インデックスを削除
        df1 = df1.reset_index(drop=True) # インデックスを削除
        df2 = df2.reset_index(drop=True) # インデックスを削除
        train_df1 = df1[df1["Probability"] >= 0.995]
        train_df2 = df2[df2["Probability"] >= 0.995]
        # 既存の訓練データと新規の予測データを結合する
        train_df1 = train_df1.rename(columns={"Prediction":target_labelname}) # 予測ラベルを正解ラベルに名称変更
        train_df2 = train_df2.rename(columns={"Prediction":target_labelname}) # 予測ラベルを正解ラベルに名称変更
        train_df1 = train_df1.drop("Probability", axis=1) # 確率を消去してラベルあり訓練データに形を合わせる
        train_df2 = train_df2.drop("Probability", axis=1) # 確率を消去してラベルあり訓練データに形を合わせる
        train_dataset1 = pd.concat([train_dataset1, train_df1], axis=0, ignore_index=True) # 訓練データセットに結合
        train_dataset2 = pd.concat([train_dataset2, train_df2], axis=0, ignore_index=True) # 訓練データセットに結合
        # モデルを学習
        model1 = learn(model1, train_dataset2, target_labelname) # モデル2で作成した教師データでモデル1を学習
        model2 = learn(model2, train_dataset1, target_labelname) # モデル1で作成した教師データでモデル2を学習
        score1 = evaluate(model1, test_dataset, target_labelname) 
        score2 = evaluate(model2, test_dataset, target_labelname)
        print(i+1, "回目: ", score1, score2)
        # 次の予測データを作成する
        next_df1 = df1[df1["Probability"] < 0.995]
        next_df2 = df2[df2["Probability"] < 0.995]
        candidate_dataset1 = next_df1.drop(["Probability", "Prediction"], axis=1) # 新たに追加した行の削除
        candidate_dataset2 = next_df2.drop(["Probability", "Prediction"], axis=1) # 新たに追加した行の削除
        candidate_dataset1 = candidate_dataset1.reset_index(drop=True) # インデックスを削除
        candidate_dataset2 = candidate_dataset2.reset_index(drop=True) # インデックスを削除
    train_dataset2.to_csv("co_train.csv")
    print("半教師あり学習のルーチンが終了しました")
    return model1, model2


def predict_probability(model, df):
    """ 予測と予測確率を算出する """
    narray = df.values
    pred = model.predict(narray)
    pred = pd.DataFrame(pred, columns=["Prediction"])
    pred_prob = model.predict_proba(narray)
    pred_prob = np.array([ps.max() for ps in pred_prob])
    pred_prob = pd.DataFrame(pred_prob, columns=["Probability"])
    df = pd.concat([df, pred, pred_prob], axis=1)
    return df


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
    prediction = model.predict(X)
    evaluation = accuracy_score(prediction, y)
    return evaluation


if __name__ == '__main__':
    # モデルのロード
    rf_filename1 = 'bst/model3.pickle'
    rf1 = joblib.load(rf_filename1)
    rf_filename2 = 'bst/model4.pickle'
    rf2 = joblib.load(rf_filename2)
    bs_filename1 = 'bst/model.pickle'
    bs1 = joblib.load(bs_filename1)
    bs_filename2 = 'bst/model2.pickle'
    bs2 = joblib.load(bs_filename2)
    
    # 訓練データのロード
    usecols = [1,2,3,4,5,6,7,9,12,13]
    names = ('RCategory','WCategory', 'RTime', 'WTime', 'AccumulatedDis', 'Velocity', 'MVelocity', 'Distance', 'Target1', 'Target2')
    filename = os.path.abspath('./') + "/training_data/training_data.csv"
    data_set = pd.read_csv(filename, sep = ",", header = None, usecols = usecols, names=names)
    data_set = data_set.sample(frac=1).reset_index(drop=True) # データをシャッフル
    train_dataset, test_dataset = data_set[:int(0.7 * len(data_set))], data_set[int(0.7 * len(data_set)):] # 訓練データとテストデータに分割
    train_dataset1 = train_dataset.drop("Target2", axis = 1) # 停止セグメント
    train_dataset2 = train_dataset.drop("Target1", axis = 1) # 活動セグメント
    test_dataset1 = test_dataset.drop("Target2", axis = 1) # 停止セグメント
    test_dataset2 = test_dataset.drop("Target1", axis = 1) # 活動セグメント

    # --- 半教師あり学習 ---
    os.chdir('../') # チェンジディレクトリ
    dir_path = os.path.abspath('./') + "/behavior_classification/training_data/"
    train_usecols = usecols[:-2] # 'Target1', 'Target2'を除去
    train_names = names[:-2] # 'Target1', 'Target2'を除去
    date = datetime.datetime(2018, 12, 20, 0, 0, 0)
    cow_id_list = get_existing_cow_list(date)
    candidate_dataset = make_features_data(dir_path, date, cow_id_list, train_usecols, train_names) # ラベルなしデータの特徴量を算出する
    #rf1 = self_semi_supervised_learning(rf1, train_dataset1, test_dataset1, candidate_dataset, "Target1", 20) # 半教師あり学習の中心的処理を行う
    #rf2 = self_semi_supervised_learning(rf2, train_dataset2, test_dataset2, candidate_dataset, "Target2", 20) # 半教師あり学習の中心的処理を行う
    rf1, bs1 = co_semi_supervised_learning(rf1, bs1, train_dataset1, test_dataset1, candidate_dataset, "Target1", 20) # 半教師あり学習の中心的処理を行う
    rf2, bs2 = co_semi_supervised_learning(rf2, bs2, train_dataset2, test_dataset2, candidate_dataset, "Target2", 20) # 半教師あり学習の中心的処理を行う

    # --- テストデータで精度の確認 ---
    print(evaluate(rf1, test_dataset1, "Target1"))
    print(evaluate(rf2, test_dataset2, "Target2"))
