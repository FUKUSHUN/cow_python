import os, sys
import csv
import datetime
import pdb

#自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import behavior_synchronization.synchronizer as synchronizer
import behavior_synchronization.community_analyzer as commnity_analyzer

# 別ファイルでモジュール化
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

def write_values(filepath, dt, value_list):
    if (os.path.exists(filepath) != True):
        with open(filepath, "w", newline='') as f: # ファイルがなければ新規作成
            writer = csv.writer(f)
            writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S")] + value_list)
    else:
        with open(filepath, "a", newline='') as f:# ファイルが存在していれば上書き
            writer = csv.writer(f)
            writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S")] + value_list)
    return

if __name__ == '__main__':
    delta_c = 5 # コミュニティの抽出間隔 [minutes]
    start = datetime.datetime(2018, 12, 1, 0, 0, 0)
    end = datetime.datetime(2018, 12, 2, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    output_file = "./behavior_synchronization/test/test.csv"
    date = start
    target_list = [20116,20192,20255,20264,20289,20295]
    #write_values(output_file, start, target_list)
    while (date < end):
        value_list = []
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        synch = synchronizer.Synchronizer(date, cow_id_list)
        analyzer = commnity_analyzer.CommunityAnalyzer(cow_id_list) # 牛のリストに更新があるため、必ずSynchronizerの後にする
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            community = synch.make_interaction_graph(t, delta_c)
            analyzer.append_community(community)
            t += datetime.timedelta(minutes=delta_c)
        # --- 1日分のコミュニティのリストを元に分析する ---
        analyzer.lookup_max_same_number()
        max_num = analyzer.get_same_num_dict() # 同一コミュニティ回数の最大値を取り出す
        for cow_id in target_list:
            value_list.append(max_num[str(cow_id)])
        write_values(output_file, date, value_list)
        date += datetime.timedelta(days=1)
