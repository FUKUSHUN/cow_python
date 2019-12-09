"""
このコードは行動分類のために行うセンサデータの前処理をまとめたコードである
"""
import numpy as np
import sys

"""
X点の移動平均（コンボリューション）を求める
Parameter
    something   : あなたが移動平均を求めたいリスト
    X   : 移動平均する点の個数 (3点であれば (t-1, t, t+1) の平均を求めることになる)
"""
def convolution(something: list, X):
    print(sys._getframe().f_code.co_name, "実行中")
    print(X, "個の移動平均をとって平滑化します---")
    b = np.ones(X) / X
    something_ave = np.convolve(something, b, 'valid')
    print("---", X, "個の移動平均をとることによる平滑化が終了しました")
    print(sys._getframe().f_code.co_name, "正常終了\n")
    return something_ave

"""
[X/2]点をリストの最初から，[X/2]点をリストの最後から消去する（移動平均による消失分）
Parameter
    something   : あなたが移動平均を求めたいリスト
    X   : 移動平均する点の個数 (3点であれば (t-1, t, t+1) の平均を求めることになる)
"""
def elimination(something: list, X):
    print(sys._getframe().f_code.co_name, "実行中")
    print(X, "個の平滑化による端点の消失分を除去します---")
    a = (int)(X / 2)
    b = len(something)- (int)(X / 2)
    something = something[a:b]
    print("---", X, "個の平滑化による端点の消失分を除去が終了しました")
    print(sys._getframe().f_code.co_name, "正常終了\n")
    return something