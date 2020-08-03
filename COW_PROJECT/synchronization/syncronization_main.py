import os, sys, gc
import csv
import datetime
import numpy as np
import pandas as pd
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

def load_change_point(filename, date:datetime.datetime):
    """ 1日分のコミュニティの変化点のリストをpd.dataframeで返す """
    start = date + datetime.timedelta(hours=9)
    end = start + datetime.timedelta(days=1)
    df = pd.read_csv(filename, header=0)
    df = df[(df.loc[:,"Start Time"]!="Start Time") | (df.loc[:,"End Time"]!="End Time")] # 特定の文字列を消去
    df = df[(start.strftime("%Y-%m-%d %H:%M:%S") <= df.loc[:,"Start Time"]) & (df.loc[:,"End Time"] <= end.strftime("%Y-%m-%d %H:%M:%S"))]
    change_point_list = df.values.tolist()
    ret = [(datetime.datetime.strptime(t[0], "%Y-%m-%d %H:%M:%S"),datetime.datetime.strptime(t[1], "%Y-%m-%d %H:%M:%S")) for t in change_point_list]
    return ret

if __name__ == '__main__':
    delta_c = 2 # コミュニティの抽出間隔 [minutes]
    delta_s = 5 # データのスライス間隔 [seconds] 
    epsilon = 12 # コミュニティ決定のパラメータ
    dzeta = 12 # コミュニティ決定のパラメータ
    leng = 5 # コミュニティ決定のパラメータ
    start = datetime.datetime(2018, 10, 1, 0, 0, 0)
    end = datetime.datetime(2018, 10, 31, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    change_point_file = "./synchronization/change_point/"
    output_file = "./synchronization/output/"
    date = start
    target_list = [20113,20170,20295,20299]
    while (date < end):
        s1 = time.time()
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        analyzer = commnity_analyzer.CommunityAnalyzer(cow_id_list) # 牛のリストに更新があるため、必ずSynchronizerの後にする
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="behavior", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, visualized_g=False, visualized_m=True, delta=delta_s, leng=leng) \
                if (t_start <= t) else [[]] # コミュニティを決定
            analyzer.append_community([t, community])
            analyzer.append_graph([t, interaction_graph])
            t += datetime.timedelta(minutes=delta_c)
        e1 = time.time()
        print("処理時間", (e1-s1)/60, "[min]")
        
        # --- 1日分のコミュニティのリストを変化点検知を行う ---
        tau, upsiron = 1, 1
        score_dict = analyzer.calculate_simpson(target_list)
        change_point_dict = analyzer.detect_change_point(target_list, tau=tau, upsiron=upsiron)
        # 結果を出力する牛のスコアのみを取り出す
        for cow_id in target_list:
            value_list = list(score_dict[str(cow_id)])
            change_point = list(change_point_dict[str(cow_id)])
            change_time_list = []
            my_utility.write_values(change_point_file+str(cow_id)+".csv", [["Start Time", "End Time"]])
            # change_point.pop(0) # 最初の要素（午前9時の牛舎にいる時間を取り除く）
            for i, start_point in enumerate(change_point):
                end_point = change_point[i+1] if (i != len(change_point)-1) else t_end
                change_time_list.append([start_point, end_point])
            my_utility.write_values(change_point_file+str(cow_id)+".csv", change_time_list)

        # --- インタラクショングラフをもとに各牛に対して分析を加える ---
        s2 = time.time()
        for cow_id in target_list:
            change_points = load_change_point(change_point_file+str(cow_id)+".csv", date)
            behavior_synch = com_creater.get_behavior_synch()
            position_synch = com_creater.get_position_synch()
            inte_analyzer = interaction_analyzer.InteractionAnalyzer(cow_id, behavior_synch, position_synch)
            # my_utility.write_values(output_file+str(cow_id)+".csv", [["Start Time", "Density Average", "Synchronization Ratio", "Walking Time", "Minimun Dist", "Minimum Dist Cow"]])
            features_list = []
            for (start_point, end_point) in change_points:
                try:
                    interval = int((end_point - start_point).total_seconds()/60)
                    community_list = analyzer.get_particular_community_list(start_point, end_point ,str(cow_id))
                    features = inte_analyzer.extract_feature(start_point, end_point, community_list, delta_c=delta_c)
                    ave_dense = analyzer.calculate_average_graph_density(start_point, end_point, str(cow_id))
                    features_list.append([start_point, ave_dense, features[2], features[0]*features[5], features[8], features[7]])
                except KeyError:
                    features_list.append([start_point, None, None, None, None, None])
            my_utility.write_values(output_file+str(cow_id)+".csv", features_list)
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        del com_creater, analyzer, inte_analyzer
        gc.collect()
        date += datetime.timedelta(days=1)
