import datetime
import sys, os
import math
import numpy as np
import pandas as pd
import pdb

class SetAnalysis:
    """ 集合同士を比較し、類似度を判定する機能を持つクラス """
    set1: set
    set2: set

    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2
        return

    def compare_set(self, rate=1/3):
        change_flag = False # 変化していればTrueにする
        eta = len(self.set1) * (1 + rate)
        theta = len(self.set1) * (1 - rate)
        if (not(len(self.set1 | self.set2) <= eta)): # not 演算をしているので注意
            change_flag = True
        if (not(len(self.set1 & self.set2) >= theta)): # not 演算をしているので注意
            change_flag = True
        return change_flag

if __name__ == "__main__":
    set1 = {1, 2, 3, 4}
    set2 = {2, 3}
    set_analyzer = SetAnalysis(set1, set2)
    result = set_analyzer.compare_set(rate=1/3)
    print(result)
