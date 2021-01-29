import os, sys
import time
import datetime
import csv
import ast
import numpy as np
import pandas as pd
import pdb

# 自作クラス
os.chdir('../../') # カレントディレクトリを二階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.functions.utility as my_utility
import synchronization.functions.plotting as plotting

def load_csv(filename):
    """ CSVファイルをロードし，記録内容をリストで返す """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        records = [row for row in reader]
    return records

def visualize_adjectory(target_cow_id, detected_cow_list, start, end, delta=5):
    s = time.time()
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    tmp_time = start - datetime.timedelta(hours=9) # 時差の分巻き戻しておく（翌朝になっている可能性もあるので）
    date = datetime.datetime(tmp_time.year, tmp_time.month, tmp_time.day, 0, 0, 0)
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
    com_creater.visualize_adjectory(start, end, [target_cow_id] + detected_cow_list, target_cow_id=target_cow_id, dirname='test/', delta=delta) # 軌跡をプロット
    print("軌跡動画と画像を作成しました: ", target_cow_id, " -> ", start.strftime('%Y/%m/%d %H:%M:%S'))
    e = time.time()
    print("処理時間", (e-s)/60, "[min]")
    return

if __name__ == "__main__":
    dirname = "./synchronization/detection_model/test/"
    cow_id = '20267'
    filepath = dirname + 'detected.csv'
    # record = pd.read_excel(filepath, usecols=[0, 1, 2, 3, 4, 5, 6])
    record = load_csv(filepath)
    for row in record:
        target_cow_id = str(row[6])
        if (target_cow_id == cow_id):
            try:
                start = datetime.datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S')
                end = datetime.datetime.strptime(row[1], '%Y/%m/%d %H:%M:%S')
            except ValueError:
                start = datetime.datetime.strptime(row[0], '%Y/%m/%d %H:%M')
                end = datetime.datetime.strptime(row[1], '%Y/%m/%d %H:%M')
            score_dic = ast.literal_eval(row[2])
            max_score = max(score_dic.values())
            detected_cow_list = []
            for key in score_dic.keys():
                if (score_dic[key] > max_score * 9/10):
                    detected_cow_list.append(key)
            visualize_adjectory(target_cow_id, detected_cow_list, start, end)
        
        

    # filepath = dirname + 'adjectory/' + filename + '.xlsx'