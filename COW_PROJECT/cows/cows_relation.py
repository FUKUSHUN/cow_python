#-*- encoding:utf-8 -*-
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import cow #自作クラス
import cowshed #自作クラス

class CowsRelation:
    main_cow_id:int #関係を見る対象の牛の個体番号
    cow_data:pd.DataFrame #牛のリスト (現時点ではpandasのデータフレーム型でidとGPS履歴を保持, main_cowも含む) 

    def __init__(self, id, df):
        self.main_cow_id = id
        self.cow_data = df
