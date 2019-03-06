#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import csv

#csv読み込み#1行目ヘッダー (移動平均後)
r_filename = "../csv/gps/weight_a.csv"
w_filename = "../csv/gps/weight_r.csv"
threthold = 3.83
df = pd.read_csv(filepath_or_buffer = r_filename, index_col = 0, encoding = "utf-8", sep = ",", header = 0)
header = ["TIME"] + list(df.head(0))
dt_list = []
#datetime型に変換
for str in df.index:
    try:
        dt_list.append(datetime.datetime.strptime(str, "%Y/%m/%d %H:%M:%S"))
    except ValueError:
        dt_list.append(datetime.datetime.strptime(str, "%Y/%m/%d"))
start = dt_list[0]
new_dt_list = []
results_list = []
while(start < dt_list[len(dt_list) - 1]):
    new_dt_list.append(start)
    end = start + datetime.timedelta(hours = 12)
    #各行に対して行う
    abnormal_scores_df = df[(start.strftime("%Y/%m/%d %H:%M:%S") <= df.index) & (df.index < end.strftime("%Y/%m/%d %H:%M:%S"))]
    is_estrus_list = [] #各牛に対してその期間発情していたか
    for i in range(len(abnormal_scores_df.columns)):
        abnormal_scores = list(abnormal_scores_df.iloc[:, i])
        is_estrus = 0
        for score in abnormal_scores:
            if(score > threthold):
                is_estrus = 1
                break
        is_estrus_list.append(is_estrus)
    results_list.append(is_estrus_list)
    start = end
with open(w_filename, mode = "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    for time, row in zip(new_dt_list, results_list):
        s = time.strftime("%Y/%m/%d %H:%M:%S")
        e = (time + datetime.timedelta(hours = 12)).strftime("%H:%M:%S")
        writer.writerow([s + "-"+ e] + row)

#csv読み込み#1行目ヘッダー (移動平均後)
r_filename = "../csv/gps/weight_r.csv"
w_filename = "../csv/gps/weight_f.csv"

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
            