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
    filename = "csv/test.csv"
    #ファイルのヘッダ書き込み
    with open(filename, mode = "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["TIME"] + list(dic.keys()))
    dt = datetime.datetime(start.year, start.month, start.day)
    a = start
    while(dt < end):
        cows = Cowshed.Cowshed(dt)
        dt = dt + datetime.timedelta(days = 1)
        with open(filename, mode = "a") as f:
            writer = csv.writer(f, lineterminator='\n')
            while(a < dt and a < end):
                df = cows.get_cow_list(a, a + datetime.timedelta(minutes = 10))
                a = a + datetime.timedelta(minutes = 10)
                partition = comm.extract_community(df, 10)
                nodes_list = []
                for com in set(partition.values()):
                    nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
                    nodes_list.append(nodes)
                    print(nodes)
                v_array = np.zeros((len(dic.values())))
                for i in range(len(df.columns)):
                    id = df.iloc[0,i]
                    data = df.iloc[1,i]
                    ana = None
                    for team in nodes_list:
                        if(id in team):
                            ana = cr.CowsRelation(id, data, team, df)
                            break
                    v_array[dic[id]] = ana.calc_evaluation_value()
                writer.writerow([a.strftime("%Y/%m/%d %H:%M:%S")] + v_array.tolist())
                del df
                gc.collect()
        del cows
        gc.collect()
        a = dt

if __name__ == '__main__':
    start = datetime.datetime(2018, 10, 1, 0, 0, 0)
    end = datetime.datetime(2018, 11, 1, 0, 0, 0)
    dic = init(end)
    main(start, end, dic)