import os, sys
import csv
import datetime
from dateutil.relativedelta import relativedelta # 1か月後を計算するため
import numpy as np
import pdb

#自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import behavior_classification.classifier as classifier

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

def choose_training_date(begin, end, N=10):
    """ 訓練に使う日付を決定する
        N: int  取り出す日付の個数 """
    date_list = []
    rang = (end - begin).days
    rnd_nums = np.arange(rang) # 0 ~ rang - 1番までの数列を作成
    np.random.shuffle(rnd_nums) # shuffleして最初のN個を取り出す
    for i in range(N):
        date = begin + datetime.timedelta(days=int(rnd_nums[i])) # 期間内のランダムな日付を取り出す
        date_list.append(date)
    return date_list

def confirm_dir(dir_path):
    """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
    if (os.path.isdir(dir_path)):
        return
    else:
        os.makedirs(dir_path)
        print("ディレクトリを作成しました", dir_path)
        return

if __name__ == '__main__':
    start = datetime.datetime(2018, 12, 1, 0, 0, 0) # 必ず1日始まりとすること
    end = start + relativedelta(months=12) # 約1か月単位
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    output_dir = "./behavior_information/"
    date = start
    month_begining = start # 月初
    while (date < end):
        # --- 月初にその月の行動の分布をまとめて学習 (事後分布の計算) ---
        if (date == month_begining):
            month_end = month_begining + relativedelta(months=1) # 翌月1日
            date_list = choose_training_date(month_begining, month_end)
            cow_id_lists = [get_existing_cow_list(choosed_date, cows_record_file) for choosed_date in date_list]
            model = classifier.Classifier()
            model.fit(date_list, cow_id_lists) # 推論
            month_begining += relativedelta(month=1) # 翌月
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        # --- 各日付, 牛ごとに分類 (予測分布を使った予測) ---
        print("行動の推論を行います: %s" %(date.strftime("%Y/%m/%d")))
        for cow_id in cow_id_list:
            output_file = output_dir + date.strftime("%Y%m%d")
            confirm_dir(output_file) # ディレクトリを作成
            t_list, v_list, labels = model.classify(date, cow_id) # 行動分類を行う
            if (len(t_list) != 0):
                try:
                    filename = output_file + "/" + str(cow_id) + ".csv"
                    model.to_csv(t_list, v_list, labels, filename) # csv出力する
                    filename = output_file + "/" + str(cow_id) + ".jpg"
                    model.plot_v_label(t_list, v_list, labels, filename)
                except:
                    print("エラーが発生しました " + date.strftime("%Y/%m/%d"))
                    pdb.set_trace()
            else:
                continue
        print("行動の推論が終了しました")
        date += datetime.timedelta(days=1)
 