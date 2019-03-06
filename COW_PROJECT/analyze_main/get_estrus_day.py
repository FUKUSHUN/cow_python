#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import csv

#csv読み込み#1行目ヘッダー (移動平均後)
r_filename = "../csv/pedometer/detect/weight_r.csv"#"../csv/gps/weight_r.csv"
w_filename = "../csv/pedometer/result/weight_f.csv"#"../csv/gps/weight_f.csv"

df = pd.read_csv(filepath_or_buffer = r_filename, index_col = 0, encoding = "utf-8", sep = ",", header = 0)
header = list(df.head(0))

estrus_detect_list = []
for i in range(len(df.columns)):
    dt = list(df.index)
    result_list = list(df.iloc[:, i])
    cow_estrus_list = []
    for day, result in zip(dt, result_list):
        if(result == 1):
            cow_estrus_list.append(day)
    estrus_detect_list.append(cow_estrus_list)

max = 0
for l in estrus_detect_list:
    if(len(l) > max):
        max = len(l)
tmp_array = np.empty((max, len(header)), dtype='U30')
for j, column in enumerate(estrus_detect_list):
    for i, r in enumerate(column):
        tmp_array[i, j] = r

estrus_detect_list = tmp_array.tolist()
with open(w_filename, mode = "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    for row in estrus_detect_list:
        writer.writerow(row)