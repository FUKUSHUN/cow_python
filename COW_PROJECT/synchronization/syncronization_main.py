import os, sys
import csv
import datetime
import numpy as np
import pdb

#自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.community_analyzer as commnity_analyzer

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

if __name__ == '__main__':
    delta_c = 5 # コミュニティの抽出間隔 [minutes]
    start = datetime.datetime(2018, 10, 21, 0, 0, 0)
    end = datetime.datetime(2018, 10, 22, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    output_file = "./synchronization/test/"
    date = start
    target_list = [20170,20295,20299]
    # write_values(output_file, [[0]+target_list])
    while (date < end):
        t_list = []
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        analyzer = commnity_analyzer.CommunityAnalyzer(cow_id_list) # 牛のリストに更新があるため、必ずSynchronizerの後にする
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=9) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            community = com_creater.make_interaction_graph(t, delta_c, method="position") if (t_start <= t) else [[]]
            analyzer.append_community([t, community])
            t += datetime.timedelta(minutes=delta_c)
        # --- 1日分のコミュニティのリストを元に分析する ---
        score_dict = analyzer.calculate_simpson(target_list)
        change_point_dict = analyzer.detect_change_point(target_list)
        # 結果を出力する牛のスコアのみを取り出す
        for cow_id in target_list:
            value_list = list(score_dict[str(cow_id)])
            change_point = list(change_point_dict[str(cow_id)])
            ###
            change_list = []
            t_tmp = date + datetime.timedelta(hours=9)
            while (t_tmp < t_end):
                change_flag = 1 if t_tmp in change_point else 0
                change_list.append(change_flag)
                t_tmp += datetime.timedelta(minutes=delta_c)
            ###
            write_values(output_file+str(cow_id)+".csv", np.array([t_list, value_list, change_list]).T.tolist())
        date += datetime.timedelta(days=1)
