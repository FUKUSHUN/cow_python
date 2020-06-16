import numpy as np
import os, sys
import pdb
from louvain import Louvain

X = np.array([[0,1,1,0,0],[1,0,1,0,0],[1,1,0,0,0],[0,0,0,0,1],[0,0,0,1,0]])
louv = Louvain()
cluster = louv.fit(X)
print(cluster)