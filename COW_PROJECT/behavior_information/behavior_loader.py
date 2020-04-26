import pandas as pd
import datetime

class BehaviorLoader:
    cow_id: int
    data: pd.DataFrame # (Time, Velocity, Behavior)

    def __init__(self, cow_id, date:datetime.datetime):
        self.cow_id = cow_id
        self._load_file(date)

    def _load_file(self, date:datetime.datetime):
        column_names = ['Time', 'Velocity', 'Behavior']
        filename = "./behavior_information/" + date.strftime("%Y%m%d/") + str(self.cow_id) + ".csv"
        df = pd.read_csv(filename, sep = ",", header = 0, usecols = [0,1,2,3], names=['index']+column_names,index_col=0) # csv読み込み
        
        # --- 1秒ごとのデータに整形し、self.dataに登録 ---
        revised_data = [] # 新規登録用リスト
        before_t = datetime.datetime.strptime(date.strftime("%Y%m%d"), "%Y%m%d") + datetime.timedelta(hours=12) # 正午12時を始まりとする
        before_v = 0.0
        before_b = 0 # 休息
        for _, row in df.iterrows():
            t = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") # datetime
            v = float(row[1])
            b = int(row[2])
            # --- 1秒ずつのデータに直し、データごとのずれを補正する ---
            while (before_t < t):
                row = (before_t, before_v, before_b)
                revised_data.append(row)
                before_t += datetime.timedelta(seconds=1)
            before_t = t
            before_v = v
            before_b = b
        end_t = datetime.datetime.strptime(date.strftime("%Y%m%d"), "%Y%m%d") + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌日午前9時を終わりとする
        while (before_t < end_t):
            t = before_t
            if (before_t < t + datetime.timedelta(seconds=5)):
                row = (before_t, before_v, before_b) # 最後のtをまだ登録していないので登録する
            else:
                row = (before_t, 0.0, 0) # 残りの時間は休息，速度ゼロで埋める
            revised_data.append(row)
            before_t += datetime.timedelta(seconds=1)
        self.data = pd.DataFrame(revised_data, columns=column_names)
    
    def get_data(self):
        return self.data