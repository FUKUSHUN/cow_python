import os, sys
import csv
import datetime
import numpy as np
import time
import pdb

#自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.community_analyzer as commnity_analyzer
import synchronization.interaction_analyzer as interaction_analyzer
import synchronization.functions.utility as my_utility

if __name__ == '__main__':
    delta_c = 2 # コミュニティの抽出間隔 [minutes]
    delta_s = 5 # データのスライス間隔 [seconds] 
    epsilon = 12 # コミュニティ決定のパラメータ
    dzeta = 10 # コミュニティ決定のパラメータ
    start = datetime.datetime(2018, 10, 18, 0, 0, 0)
    end = datetime.datetime(2018, 10, 24, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    output_file = "./synchronization/change_point/"
    date = start
    target_list = [20113,20170,20295,20299]
    s = time.time()
    while (date < end):
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        analyzer = commnity_analyzer.CommunityAnalyzer(cow_id_list) # 牛のリストに更新があるため、必ずSynchronizerの後にする
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=9) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="behavior", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, visualized_g=False, visualized_m=False, delta=delta_s) \
                if (t_start <= t) else [[]] # コミュニティを決定
            analyzer.append_community([t, community])
            analyzer.append_graph([t, interaction_graph])
            t += datetime.timedelta(minutes=delta_c)
        e = time.time()
        print("処理時間", (e-s)/60, "[min]")
        # --- 1日分のコミュニティのリストを変化点検知を行う ---
        tau, upsiron = 1, 1
        score_dict = analyzer.calculate_simpson(target_list)
        change_point_dict = analyzer.detect_change_point(target_list, tau=tau, upsiron=upsiron)
        # 結果を出力する牛のスコアのみを取り出す
        for cow_id in target_list:
            value_list = list(score_dict[str(cow_id)])
            change_point = list(change_point_dict[str(cow_id)])
            # behavior_synch = com_creater.get_behavior_synch()
            # position_synch = com_creater.get_position_synch()
            # inte_analyzer = interaction_analyzer.InteractionAnalyzer(cow_id, behavior_synch, position_synch)
            change_time_list = []
            my_utility.write_values(output_file+str(cow_id)+".csv", [["Start Time", "End Time"]])
            for i, start_point in enumerate(change_point):
                if (i != 0):
                    end_point = change_point[i+1] if (i != len(change_point)-1) else t_end
                    # interval = int((end_point - start_point).total_seconds()/60)
                    # features = inte_analyzer.extract_feature(start_point, end_point, community)
                    change_time_list.append([start_point, end_point])
            my_utility.write_values(output_file+str(cow_id)+".csv", change_time_list)
        date += datetime.timedelta(days=1)
