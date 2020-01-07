#-*- encoding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
import datetime
import os

class ApproachedIndexWriter:
    _writingfilename = "./for_web/"
    cow_id = None
    def __init__(self, w_filename, cow_id):
        self._writingfilename = w_filename
        self.cow_id = cow_id
    
    def write_values(self, dt, value):
        if (os.path.exists(self._writingfilename) != True):
            with open(self._writingfilename, "w", newline='') as f: # ファイルがなければ新規作成
                writer = csv.writer(f)
                writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S"), value])
        else:
            with open(self._writingfilename, "a", newline='') as f:# ファイルが存在していれば上書き
                writer = csv.writer(f)
                writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S"), value])

class ApproachedValueWriter:
    _readingfilename = "./for_web/"
    _writingfilename = "./for_web/"
    cow_id = None
    def __init__(self, r_filename, w_filename, cow_id):
        self._readingfilename = r_filename
        self._writingfilename = w_filename
        self.cow_id = cow_id

    def calc_approached_value(self, dt, count):
        values = self._read_values(dt, count)
        approached_value = sum(values) / len(values) if (len(values) != 0) else 0
        self._write_values(dt, approached_value)

    def _read_values(self, dt, count):
        """ 被接近指標を補完しているファイルから該当時間の値を取得する """
        data_list = [] #被接近指標を格納するリスト
        start_time = dt - datetime.timedelta(minutes=count) # 時間を戻す
        with open(self._readingfilename, mode = "r", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                time = datetime.datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S")
                if (start_time < time and time <= dt):
                    data_list.append(float(row[1]))
        return data_list

    def _write_values(self, dt, value):
        """ 接近度をファイルに書き込む """
        if (os.path.exists(self._writingfilename) != True):
            with open(self._writingfilename, "w", newline='') as f: # ファイルがなければ新規作成
                writer = csv.writer(f)
                writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S"), value])
        else:
            with open(self._writingfilename, "a", newline='') as f:# ファイルが存在していれば上書き
                writer = csv.writer(f)
                writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S"), value])
        return