#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import cows.cows_community_rssi as comm
import cows.cows_relation_rssi as cr
import cows.cowshed_rssi as Cowshed
import datetime
import gc
import csv
import time
import os

"""
被接近度（コサイン類似度）を算出するプログラム
ここでは移動平均を算出する前のファイルを出力するので，移動平均を出すにはanalyze_main/move_ave.pyを実行すること
"""

def main_process(start, end):
    """ 被接近度を算出するメインプロセス
        startとendの間隔は1時間であることが約束されている 
        dictを用意し，10分ごとの被接近度を各牛の個体番号の番地に追加していく
        (イメージ)
        {20xxx:[0.653, 0.134, ...], 20yyy:[0.415, 0.345, ...], ...}
        """
    dt = start
    cows = Cowshed.Cowshed(dt) # その時いる牛を取得
    cow_ids = cows.get_cow_ids() # その時いる牛の個体番号を取得
    dic = {cow_id: [] for cow_id in cow_ids} # すべての牛に対して個体番号に対応したリストを持つ辞書を作成．初期状態はすべてリストは空

    while(dt < end):
        print(dt.strftime("%Y/%m/%d %H:%M:%S"))
        df = cows.get_cow_list(dt, dt + datetime.timedelta(minutes = 10)) # 10分ごとに算出
        nodes_list = comm.extract_community(df, 10) #コミュニティ生成
        for i in range(len(df.columns)):
            cow_id = df.iloc[0,i] # 牛iの個体番号
            data = df.iloc[1,i] # 牛iの位置情報
            # 牛iの位置情報がなければbreak
            if (data is None):
                break
            else:
                # コミュニティの中から牛iがいるコミュニティを見つける
                for team in nodes_list:
                    if(cow_id in team):
                        relation = cr.CowsRelation(cow_id, dt, data, team, df)
                        approached_value = relation.calc_evaluation_value()
                        if (approached_value != 0): # 現在は通信量の関係で10分ごとに空データが入るので値が0になる
                            dic[cow_id].append(approached_value)
                        break
        del df
        dt = dt + datetime.timedelta(minutes = 10) # 10分進む
    del cows
    gc.collect()
    output_to_csv(start, dic)

def output_to_csv(dt, dic):
    """ 辞書型の結果
         (イメージ)
        {20xxx:[0.653, 0.134, ...], 20yyy:[0.415, 0.345, ...], ...}
        を引数として受け取り，各個体番号から指定のフォルダに20xxx.csvという結果のファイルを出力する 
        Parameter
            dt  : 評価値の時刻
            dic : その時刻の牛ごとの評価値辞書 """
    
    filepath = "./for_web/"
    # キーを取得しid.csvのファイルがなければ生成
    for mykey in dic.keys():
        myfilename = filepath + str(mykey) + ".csv" # 書き込みファイル名
        if (os.path.exists(myfilename) != True):
            with open(myfilename, "w") as f: # ファイルがなければ新規作成
                value = dic[mykey]
                if (len(value) == 0):
                    score = 0
                    f.write(dt.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(score) + "\n")
                else:
                    score = sum(value) / len (value)
                    f.write(dt.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(score) + "\n")
        else:
            with open(myfilename, "a") as f:# ファイルが存在していれば上書き
                value = dic[mykey]
                if (len(value) == 0):
                    score = 0
                    f.write(dt.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(score) + "\n")
                else:
                    score = sum(value) / len (value)
                    f.write(dt.strftime("%Y-%m-%dT%H:%M:%S") + "," + str(score) + "\n")
    return


if __name__ == '__main__':
    start = datetime.datetime(2018, 10, 19, 0, 0, 0)
    end = datetime.datetime(2018, 10, 25, 0, 0, 0)
    s = time.time()
    while (start < end):
        main_process(start, start + datetime.timedelta(hours=1))
        start += datetime.timedelta(hours=1)
    e = time.time()
    print("実行時間: ", e - s, "秒")