#-*- encoding:utf-8 -*-
import sys
import statistics
import pandas as pd
import numpy as np
import math

class MyGaussianDistribution:
    df = None
    mean_vector = None
    cov_matrix = None

    def __init__(self, df):
        self.df = df
        self.mean_vector = self.calc_mean_vector(df)
        self.cov_matrix = self.calc_cov_matrix(df)

    def calc_mean_vector(self, df):
        """ 平均ベクトルを求める """
        mean_vector = np.zeros((len(df.columns), 1))
        for i, (_, column) in enumerate(df.iteritems()):
            mean_vector[i, 0] = statistics.mean(column)
        print(mean_vector)        
        return mean_vector

    def calc_cov_matrix(self, df):
        """ 共分散行列を求める """
        cov_matrix = np.zeros((len(df.columns), len(df.columns)))
        for i, (_, column1) in enumerate(df.iteritems()):
            for j, (_, column2) in enumerate(df.iteritems()):
                x_list, y_list = column1.values.tolist(), column2.values.tolist()
                cov_matrix[i, j] = self._get_cov(x_list, y_list)
        print(cov_matrix)
        return cov_matrix


    def _get_cov(self, x_list, y_list):
        """ 共分散を求める """
        if (len(x_list) != len(y_list)):
            print(x_list, "と", y_list, "の長さが異なります")
            sys.exit()
        else:
            ns_xy = 0.0
            n = len(x_list) # 標本数
            x_ave = statistics.mean(x_list)
            y_ave = statistics.mean(y_list)
            for x, y in zip(x_list, y_list):
                ns_xy += (x - x_ave) * (y - y_ave)
            s_xy = ns_xy / n
            return s_xy

    def get_mahalanobis_distance(self, x:np.array):
        """ マハラノビス距離 (1変数のときの偏差値みたいなもの) を求める 
            Paramete
                x   :ndarray """
        if (len(x) != len(self.mean_vector)):
            print(sys._getframe().f_code.co_name)
            print("引数のベクトル長が不正です")
        else:
            inv_cov_matrix = np.linalg.inv(self.cov_matrix) # 逆行列
            dis2 = np.dot(np.dot((x - self.mean_vector).T,  inv_cov_matrix), (x - self.mean_vector))
            dis = math.sqrt(dis2)
            return dis


    def get_mean_vector(self):
        """ 平均ベクトルを取得する """
        return self.mean_vector

    
    def get_cov_matrix(self):
        """ 共分散行列を取得する """
        return self.cov_matrix


"""
if __name__ == "__main__":
    rest_dataset_file = "./training_data/rest_train_data.csv"
    df = pd.read_csv(rest_dataset_file, sep = ",", header = 0, usecols = [0,3,5,6], names=('Time', 'RTime', 'AccumulatedDis', 'VelocityAve')) # csv読み込み
    used_df = df.drop('Time', axis=1)
    mean_vector = calc_mean_vector(used_df)
    print(mean_vector)
    cov_matrix = calc_cov_matrix(used_df)
    print(cov_matrix)
    #plotting.show_3d_plot(df['RTime'], df['AccumulatedDis'], df['VelocityAve'])
    inv_cov_matrix = np.linalg.inv(cov_matrix) # 逆行列
    #print(inv_cov_matrix)
    #print(np.dot(cov_matrix, inv_cov_matrix))
"""
