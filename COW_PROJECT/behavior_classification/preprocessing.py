"""
このコードは行動分類のために行うセンサデータの前処理をまとめたコードである
"""
import numpy as np

"""
X点の移動平均（コンボリューション）を求める
Parameter
    something   : あなたが移動平均を求めたいリスト
    X   : 移動平均する点の個数 (3点であれば (t-1, t, t+1) の平均を求めることになる)
"""
def convolution(something: list, X):
    b = np.ones(X) / X
    something_ave = np.convolve(something, b, 'valid')
    return something_ave

"""
[X/2]点をリストの最初から，[X/2]点をリストの最後から消去する（移動平均による消失分）
Parameter
    something   : あなたが移動平均を求めたいリスト
    X   : 移動平均する点の個数 (3点であれば (t-1, t, t+1) の平均を求めることになる)
"""
def elimination(something: list, X):
    a = (int)(X / 2)
    b = len(something)- (int)(X / 2)
    something = something[a:b]
    return something