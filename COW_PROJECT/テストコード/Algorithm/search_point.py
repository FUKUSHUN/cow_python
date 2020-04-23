import numpy as np
import random
import pdb

""" 分岐点を探索するアルゴリズムのテストコード
    ソートした数値を小さい方から分岐していき重心間距離の2乗の総和が最小となる分岐点を (ほぼ) 全数探索する """

X = []
# 乱数を100個生成
for i in range(100):
    X.append(random.randint(1, 300))

Y = X
X = sorted(X)

X1 = X[0:2]
X2 = X[2:]

min_d = None
min_i = 2

for i in range(2, len(X)-2):
    x1_g = sum(X1) / len(X1) # 重心
    x2_g = sum(X2) / len(X2) # 重心
    d = 0
    for x in X1:
        d += (x - x1_g) ** 2
    for x in X2:
        d += (x - x2_g) ** 2
    # 最小値の更新
    if (min_d is None or d < min_d):
        min_d = d
        min_i = i
    X1 = X[0:i+1]
    X2 = X[i+1:]
threshold = (X[min_i-1] + X[min_i]) / 2

pdb.set_trace()

