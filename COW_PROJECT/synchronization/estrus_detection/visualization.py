import os, sys
import numpy as np
import pandas as pd
import pdb

# 自作クラス
from plotting import PlotMaker3D

def calculate_dist_params(arrays):
    """ 平均のベクトルと分散共分散行列を求める """
    mu = np.mean(arrays, axis=0)
    cov = np.cov(arrays, rowvar=0, bias=0)
    return mu, cov

def detect_anomaly(vec, mu, cov, threshold=7.81):
    """ 異常値かどうかを判定する """
    mu = np.reshape(mu, [len(mu),1])
    lam = np.linalg.inv(cov) # 精度行列
    vec = np.reshape(vec, [len(vec),1])
    a = np.dot((vec - mu).T, np.dot(lam, (vec - mu)))
    if (a > threshold):
        if (vec[0] > mu[0] and vec[1] > mu[1]):
            print("異常度: ", a, ", ベクトル: ", vec)
            return True
    else:
        return False


if __name__ == "__main__":
    filenames1 = ['20299_1.csv']
    filenames2 = ['20299.csv']
    for filename1, filename2 in zip(filenames1, filenames2):
        # csvファイルをロードする
        df = pd.read_csv(filename2, engine='python', names=['time', 'dence', 'isolation', 'non-rest', 'interval'])
        walk_time = df['non-rest'].values * df['interval'].values
        df = pd.concat([df, pd.Series(walk_time, name='walk_time')], axis=1) # 歩行時間を追加してdfを整形
        mu, cov = calculate_dist_params(df.loc[:, ['dence', 'isolation', 'interval']].values) # 平均と共分散行列を求める

        df2 = pd.read_csv(filename1, engine='python', names=['time', 'dence', 'isolation', 'non-rest', 'interval'])
        walk_time = df2['non-rest'].values * df2['interval'].values
        df2 = pd.concat([df2, pd.Series(walk_time, name='walk_time')], axis=1) # 歩行時間を追加してdfを整形
        # 異常値を検出し，出力する
        with open(filename1[:-4] + '.txt', 'w') as f:
            for i, vec in enumerate(df2.loc[:, ['dence', 'isolation', 'interval']].values):
                judge = detect_anomaly(vec, mu, cov, threshold=7.81)
                if (judge):
                    print(df2.loc[i, :], file=f)