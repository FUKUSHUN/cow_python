#-*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import shutil
import cows.cows_community as comm
import cows.cows_relation as cr
import cows.cowshed as Cowshed
import datetime
import gc
import csv
import time
"""
被接近度（コサイン類似度）を算出するプログラム
ここでは移動平均を算出する前のファイルを出力するので，移動平均を出すにはanalyze_main/move_ave.pyを実行すること
"""
def init(dt):
    #シリアライズファイルの初期化
    shutil.rmtree("serial")
    os.mkdir("serial")
    #存在リストの確認
    path = "../CowTagOutput/Record/PastExistRecord.csv"
    with open(path, mode = "r") as f:
        reader = csv.reader(f)
        cow_list = []
        for row in reader:
            date = row[0]
            if(dt < datetime.datetime.strptime(date, "%Y/%m/%d")):
                cow_list = row[1:]
    #各牛に列番号を割り振り
    dic = {}
    for i, num in enumerate(cow_list):
        dic[int(num)] = i
    return dic

def main(start, end, dic):
    print(dic)
    filename = "for_web/weight_gps.csv"
    fheaderwrite(filename, list(dic.keys()))
    dt = datetime.datetime(start.year, start.month, start.day)
    a = start
    while(dt < end):
        cows = Cowshed.Cowshed(dt)
        dt = dt + datetime.timedelta(days = 1)
        data_list = []
        while(a < dt and a < end):
            print(a.strftime("%Y/%m/%d %H:%M:%S"))
            df = cows.get_cow_list(a, a + datetime.timedelta(minutes = 10))
            a = a + datetime.timedelta(minutes = 10)
            nodes_list = comm.extract_community(df, 10) #コミュニティ生成
            v_array = np.zeros((len(dic.values()))) #各評価値を行ベクトルの形式で保存するので全ての要素0で初期化
            for i in range(len(df.columns)):
                id = df.iloc[0,i]
                data = df.iloc[1,i]
                ana = None
                for team in nodes_list:
                    if(id in team):
                        ana = cr.CowsRelation(id, data, team, df)
                        break
                v_array[dic[id]] = ana.calc_evaluation_value()
            data_list.append([a.strftime("%Y/%m/%d %H:%M:%S")] + v_array.tolist())
            del df
            gc.collect()
        fwrite(filename, data_list) #1日ずつファイルに書き込む (もしかしたら最後に一気に書き込んだ方が良いかもしれない)
        del cows
        gc.collect()
        a = dt
        
#ファイルのヘッダ書き込み
def fheaderwrite(filename, member_list):
    with open(filename, mode = "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["TIME"] + member_list)

#1日分のデータの書き込み 
def fwrite(filename, data):
    with open(filename, mode = "a") as f:
        writer = csv.writer(f, lineterminator='\n')
        for row in data:
            writer.writerow(row)
    return

if __name__ == '__main__':
    start = datetime.datetime(2018, 10, 19, 0, 0, 0)
    end = datetime.datetime(2018, 10, 26, 0, 0, 0)
    s = time.time()
    dic = init(end)
    main(start, end, dic)
    e = time.time()
    print(e - s)