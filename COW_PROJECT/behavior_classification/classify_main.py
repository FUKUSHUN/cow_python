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
    return

def confirm_dir(dir_path):
    """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
    if (os.path.isdir(dir_path)):
        return
    else:
        os.makedirs(dir_path)
        print("ディレクトリを作成しました", dir_path)
        return

if __name__ == '__main__':
    start = datetime.datetime(2018, 12, 30, 0, 0, 0)
    end = datetime.datetime(2018, 12, 31, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    output_dir = "./behavior_information/"
    date = start
    while (date < end):
        cow_id_list = get_existing_cow_list(date, cows_record_file)
        for cow_id in cow_id_list:
            output_file = output_dir + date.strftime("%Y%m%d")
            confirm_dir(output_file) # ディレクトリを作成
            model = classifier.Classifier()
            t_list, v_list, labels = model.classify(date, cow_id) # 行動分類を行う
            if (len(t_list) != 0):
                filename = output_file + "/" + str(cow_id) + ".csv"
                model.to_csv(t_list, v_list, labels, filename) # csv出力する
                filename = output_file + "/" + str(cow_id) + ".jpg"
                model.plot_v_label(t_list, v_list, labels, filename)
            else:
                continue
        date += datetime.timedelta(days=1)
 