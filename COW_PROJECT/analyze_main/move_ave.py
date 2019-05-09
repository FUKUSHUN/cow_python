#-*- encoding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
import datetime

def read_values(filepath):
    count = 18 #何個分の移動平均をとるか
    df = pd.read_csv(filepath, header = 0, index_col=0)
    header = ["TIME"] + list(df.head(0))
    df = df.fillna(0) #nanを0に
    index_list = df.index[1 * (count - 1):]
    data_list = [] #各列ごとの移動平均のリストのリスト
    for i in range(len(df.columns)):
        cosines = df.iloc[:, i]
        data_list.append(convo_per_count(cosines, count))
    tmp_array = np.array(data_list) #行列の転置を行う
    data_list = (tmp_array.T).tolist()
    return header, index_list, data_list

def convo_per_count(values, count):
    value_list = []
    new_value_list = []
    for data in values:
        value_list.append(data)
        if(len(value_list) >= count):
            value_list = value_list[(-1 * count):] #pop(0)
            new_value_list.append(sum(value_list) / count)
    return new_value_list

def write_values(filename, header, index_list, data_list):
    with open(filename, mode = "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for time, row in zip(index_list, data_list):
            time_dt = datetime.datetime.strptime(time, "%Y/%m/%d %H:%M:%S") + datetime.timedelta(hours = 9)
            time = time_dt.strftime("%Y/%m/%d %H:%M:%S")
            writer.writerow([time] + row)
    return

if __name__ == '__main__':
    r_filename = "csv/weight.csv"
    w_filename = "csv/weight_m.csv"
    header, times, values = read_values(r_filename)
    write_values(w_filename, header, times, values)