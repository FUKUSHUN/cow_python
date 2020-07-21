import os, sys
import csv
import datetime
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

if __name__ == '__main__':
    start = datetime.datetime(2018, 9, 26, 0, 0, 0)
    end = datetime.datetime(2018, 10, 1, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    date = start
    while (date < end):
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        for cow_id in cow_id_list:
            model = classifier.Classifier()
            model.classify(date, cow_id) # 行動分類結果をcsvファイルに出力する
        date += datetime.timedelta(days=1)
 