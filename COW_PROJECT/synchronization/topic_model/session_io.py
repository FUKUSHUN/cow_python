import os, sys
import csv
import glob
import datetime
import numpy as np
import pandas as pd
import pdb
import pickle

""" セッションを一時的に保管する際に使うファイルI/O """

def write_session(session_list, dirpath):
    """ セッションの書き込み．1ファイル1セッション """
    for i, session in enumerate(session_list):
        _confirm_dir(dirpath)
        filename = dirpath+ str(i)
        with open(filename +".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for w in session:
                writer.writerow(w)
    return

def read_session(dirpath):
    """ セッションの読み込み．ディレクトリ直下のcsvファイルすべて """
    corpus = []
    files = glob.glob(dirpath + "*.csv")
    for filename in files:
        with open(filename) as f:
            reader = csv.reader(f)
            corpus.append(reader)
    return np.array(corpus)

def _confirm_dir(dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return