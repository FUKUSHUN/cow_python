import os,sys
import datetime
import pdb

# 自作クラス
os.chdir('../../') # カレントディレクトリを./COW_PROJECT/へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import テストコード.Visualization.image.adjectory_image as adjectory
import cows.cowshed as cowshed

if __name__ == '__main__':
    start = datetime.datetime(2018, 12, 30, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)
    end = datetime.datetime(2018, 12, 31, 0, 0, 0) # イギリス時間 (時差9時間なのでちょうど良い)

    the_day = start
    cows = cowshed.Cowshed(the_day) # その日の牛の集合
    dt = start
    next_dt = dt
    while (dt < end):
        next_dt += datetime.timedelta(hours=1)
        cow_df = cows.get_cow_list(dt, next_dt)
        # 時間と牛のIDを元にした位置情報のマトリックスを作る (欠損に対応するため，短い間隔でつくっていく)
                
        dt = next_dt # 時間を進める