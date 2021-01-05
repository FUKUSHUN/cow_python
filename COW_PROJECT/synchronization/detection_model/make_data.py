import os, sys
import csv
import datetime
import numpy as np
import pandas as pd
import json
import pdb

# 自作クラス
os.chdir('../../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater

# 自作メソッド
import cows.geography as geography
import synchronization.functions.utility as my_utility

""" 
発情期のインタラクションを検出するための入力データを作成する (main関数) 
"""

def load_csv(filename):
    """ CSVファイルをロードし，記録内容をリストで返す """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        records = [row for row in reader]
    return records

def fetch_information(date):
    """ 1日分の牛のデータを読み込む """
    date = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
    cow_id_list = com_creater.cow_id_list
    beh_synch, pos_synch = com_creater.get_behavior_synch(), com_creater.get_position_synch()
    return cow_id_list, beh_synch, pos_synch

def extract_information(beh_synch, pos_synch, start, end):
    """ startからendまでの時間のデータをdeltaごとにスライスして抽出し，行動，空間の2つのデータを得る(どちらも1秒ごとに成形し，インデックスがTimeになっている前提) """
    delta = 5 # データのスライス間隔 [seconds]
    beh_df = beh_synch.extract_df(start, end, delta)
    pos_df = pos_synch.extract_df(start, end, delta)
    return beh_df, pos_df

def scan_data(beh_df, pos_df, target_cow_id, start, end, judge):
    """ データを走査し近い距離にいた時間が長い順に牛をn頭取り出した後，牛集合の2分ごとのデータを作成する
        judge:  出力ラベル """
    delta = 2 # データのスキャン間隔（コミュニティの作成間隔）[minutes]
    distance_th = 10 # 近い距離の定義
    target_cow_id = str(target_cow_id)
    scna_time = start
    scan_data = []
    near_cows = _rank_cow(pos_df, target_cow_id, threshold=distance_th)
    while (scna_time < end):
        next_time = scna_time + datetime.timedelta(minutes=delta)
        beh_df2 = beh_df[(scna_time <= beh_df.index) & (beh_df.index < next_time)] # 抽出するときは代わりの変数を用意すること
        pos_df2 = pos_df[(scna_time <= pos_df.index) & (pos_df.index < next_time)] # 抽出するときは代わりの変数を用意すること
        beh_df2 = beh_df2[[str(target_cow_id)] + near_cows]
        pos_df2 = pos_df2[[str(target_cow_id)] + near_cows]
        input_image = _orgnize_matrix(beh_df2, pos_df2, target_cow_id, near_cows, threshold=distance_th)
        output = [judge]
        data = {'Time': scna_time.strftime("%Y/%m/%d %H:%M:%S"), 'Input': input_image.tolist(), 'Output': output, 'Target': target_cow_id}
        scan_data.append(data)
        scna_time = next_time
    return scan_data

def _rank_cow(pos_df, target_cow_id, threshold=10):
    """ 観測対象牛と閾値以内の距離にいた時間が多い順にx個の牛を取り出す (抽出後のdfは降順には並び変えない)
        threshold:  距離の近さの閾値 [meter] """
    num_extracted = 5 # 取り出す牛の頭数
    near_cows = {}
    cow_id_list = list(pos_df.columns)
    cow_id_list.remove(target_cow_id) # 観測対象牛は除去する
    target_pos = pos_df[target_cow_id].values
    for cow_id in cow_id_list:
        nearcow_pos = pos_df[cow_id].values
        count = 0
        for t_p, n_p in zip(target_pos, nearcow_pos):
            lat1, lon1, lat2, lon2 = t_p[0], t_p[1], n_p[0], n_p[1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            if (dis <= threshold):
                count += 1 # 近い距離にいた回数をカウント
        near_cows[cow_id] = count
    sorted_near_cows = sorted(near_cows.items(), key=lambda x:x[1], reverse=True) # 近い距離にいた時間が長い牛からソート
    near_cows = sorted([elem[0] for elem in sorted_near_cows[:num_extracted]]) # 上位x個を抽出し牛のIDのみをリストに格納する
    return near_cows

def _orgnize_matrix(beh_df, pos_df, target_cow_id, near_cows, threshold=10):
    """ 入力形式に従ってmatrixを作成する
        threshold:  距離の近さの閾値 [meter] """
    target_beh = beh_df[str(target_cow_id)].values
    target_pos = pos_df[str(target_cow_id)].values
    input_image = np.zeros((len(target_beh) * 3, len(near_cows) * 3))
    i = 0
    for cow_id in near_cows:
        nearcow_beh = beh_df[cow_id].values
        nearcow_pos = pos_df[cow_id].values
        j = 0
        for t_b, t_p, n_b, n_p in zip(target_beh, target_pos, nearcow_beh, nearcow_pos):
            lat1, lon1, lat2, lon2 = t_p[0], t_p[1], n_p[0], n_p[1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            if (dis <= threshold):
                mat = np.zeros((3, 3)) # 3 * 3のマトリックスに行動を埋め込む
                mat[t_b, n_b] = 1
            else:
                mat = np.zeros((3, 3)) # 近い距離にいない場合はゼロ行列
            input_image[j*3:(j+1)*3, i*3:(i+1)*3] = mat # 対応するフィルターに対して行列を代入する
            j += 1
        i += 1
    return input_image

if __name__ == "__main__":
    recordfile = './synchronization/detection_model/record.csv'
    records = load_csv(recordfile)
    recordjsonfile = './synchronization/detection_model/data.json'
    json_data = {}
    my_utility.delete_file(recordjsonfile) # jsonファイルを書き込むときは一度ファイルを消去する
    for i, row in enumerate(records):
        date = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S") - datetime.timedelta(hours=9) # 翌朝の可能性もあるので時間を9時間戻す
        cow_id_list, beh_synch, pos_synch = fetch_information(date)
        start = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S")
        end = datetime.datetime.strptime(row[2] + " " + row[3], "%Y/%m/%d %H:%M:%S") + datetime.timedelta(seconds=5) # 不等号で抽出するため5秒追加
        beh_df, pos_df = extract_information(beh_synch, pos_synch, start, end)
        data = scan_data(beh_df, pos_df, row[4][:5], start, end, int(row[5]))
        json_data[str(i)] = data
        my_utility.write_json(recordjsonfile, json_data)

