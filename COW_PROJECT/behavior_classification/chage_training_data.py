import output_features
import pandas as pd
import datetime

def get_time(series):
    rest_list = []
    walk_list = []
    for i, string in series.iteritems():
        times = string.split("|")
        rest_start = datetime.datetime.strptime(times[0].split("-")[0], "%Y/%m/%d %H:%M:%S")
        walk_start = datetime.datetime.strptime(times[1].split("-")[0][1:], "%Y/%m/%d %H:%M:%S")
        rest_catego = output_features.decide_time_category(rest_start, rest_start)
        walk_catego = output_features.decide_time_category(walk_start, walk_start)
        rest_list.append(rest_catego)
        walk_list.append(walk_catego)
    rest_series = pd.Series(rest_list)
    walk_series = pd.Series(walk_list)
    df = pd.concat([series, rest_series, walk_series], axis = 1, ignore_index=True)
    df.to_csv("./training_data/category.csv")


if __name__ == '__main__':
    filename = "./training_data/training_data.csv"
    df = pd.read_csv(filename, sep = ",", header = None, usecols = [0,1,2], names=('Time', 'RCategory', 'WCategory')) # csv読み込み
    get_time(df['Time'])

