"""
このコードは文字列操作について独自に必要な処理を関数化したものをまとめている
"""
import datetime

"""
時系列データ (start-end) | (start-end) をdatetime型にして [[t1, t2], [t3, t4], [...], [...], ...] のような形にする
例) 2018/12/30 13:00:30-2018/12/30 13:00:30 | 2018/12/30 13:00:35-2018/12/30 13:00:45
Parameter
    strt_list  : 上のような規則で書かれた文字列が入ったリスト
"""
def str_to_datetime(strt_list):
    dtt_list = []
    for string in strt_list:
        former, latter = string.split(" | ")
        start_f, end_f = former.split("-")
        start_l, end_l = latter.split("-")
        dtt_list.append([datetime.datetime.strptime(start_f, "%Y/%m/%d %H:%M:%S"), datetime.datetime.strptime(end_f, "%Y/%m/%d %H:%M:%S")])
        dtt_list.append([datetime.datetime.strptime(start_l, "%Y/%m/%d %H:%M:%S"), datetime.datetime.strptime(end_l, "%Y/%m/%d %H:%M:%S")])
    return dtt_list
