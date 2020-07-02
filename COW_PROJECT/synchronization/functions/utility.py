import os, sys
import datetime
import csv

""" 便利系のファンクションを格納（共通して使用するがまだくくりがないものを集める。メイン関数で使用される） """

def get_existing_cow_list(date:datetime, filepath):
    """ 引数の日にちに第一放牧場にいた牛のリストを得る """
    filepath = filepath + date.strftime("%Y-%m") + ".csv"
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if (datetime.datetime.strptime(row[0], "%Y/%m/%d") == date):
                return row[1:]
    print("指定の日付の牛のリストが見つかりません", date.strftime("%Y/%m/%d"))
    sys.exit()

def write_values(filepath, value_list):
    if (os.path.exists(filepath) != True):
        with open(filepath, "w", newline='') as f: # ファイルがなければ新規作成
            writer = csv.writer(f)
            writer.writerows(value_list)
    else:
        with open(filepath, "a", newline='') as f:# ファイルが存在していれば上書き
            writer = csv.writer(f)
            writer.writerows(value_list)
    return