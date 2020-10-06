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

    def compare_set(self):
        change_flag = False # 変化していればTrueにする
        return change_flag