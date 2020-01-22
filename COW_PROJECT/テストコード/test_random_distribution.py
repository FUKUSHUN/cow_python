import numpy as np
from scipy import stats

#カテゴリカル分布（歪んだサイコロ）
xk = np.arange(6)
pk = np.array([1/6,1/6,1/6,1/6,1/6,1/6]) #仮に1を超えた場合はそれ以降の値は出現しない（この場合は0~3までが生成される）
custm = stats.rv_discrete(name='custm', values=(xk, pk))
print(custm.rvs(size=1))

# ガンマ分布
print(np.random.gamma(1, scale=1/5))

# ディリクレ分布
print(np.random.dirichlet(np.array([0.5, 0.5, 0.5])))

# ウィシャート分布
wish = stats.wishart(df=3,scale=np.array([[1.0, 0.5], [0.5, 1.0]]))
print(wish.rvs(1)[0]) # 乱数を1個生成

# ガウス分布
print(np.random.multivariate_normal(np.array([0, 0]), np.array([[1.0, 0.5], [0.5, 2.0]])))