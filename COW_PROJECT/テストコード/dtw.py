import numpy as np
import pylab
import matplotlib.gridspec as gridspec

d = lambda a,b:abs(a - b) # 距離
first = lambda x: x[0] # 配列の1番目の要素
second = lambda x: x[1] # 配列の2番目の要素

def minimum_cost(v1, v2, v3):
    minimum = min(first(v1), first(v2), first(v3))
    if ((first(v1) == minimum) and (first(v1) != -2)):
        return v1, 0
    elif ((first(v2) == minimum) and (first(v2) != -2)):
        return v2, 1
    else:
        return v3, 2


def calc_dtw(A, B):
    I = len(A)
    J = len(B)
    W = 12 # 整合窓（これ以上離れているデータとの類似性は見ない）

    f = [[0 for j in range(J)] for i in range(I)]
    f[0][0] = (d(A[0],B[0]), (-1,-1))
    for i in range(1,I):
            f[i][0] = (f[i-1][0][0] + d(A[i], B[0]), (i-1,0)) if (abs(i - 0) <=W) else (-2, (i-1,0))
    for j in range(1,J):
            f[0][j] = (f[0][j-1][0] + d(A[0], B[j]), (0,j-1)) if (abs(0 - j) <=W) else (-2, (0,j-1))
    
    for i in range(1,I):
        for j in range(1,J):
            if (abs(i - j) <= W):
                minimum, index = minimum_cost(f[i-1][j], f[i][j-1], f[i-1][j-1])
                indexes = [(i-1,j), (i,j-1), (i-1,j-1)]
                if (index == 2):
                    f[i][j] = (first(minimum) + d(A[i], B[j]) * 2, indexes[index])
                else:
                    f[i][j] = (first(minimum) + d(A[i], B[j]) * 1, indexes[index])
            else:
                f[i][j] = (-2, (i-1, j-1)) # 関係のないデータは-2で区別
    return f


def get_path(f):
    """ 経路を求める """
    path = [[len(f)-1, len(f[0])-1]]
    while(True):
        path.append(f[path[-1][0]][path[-1][1]][1])
        if(path[-1]==(0,0)):
            break
    path = np.array(path)
    print(path)
    return path


def plot_path(f, path, A, B):
    gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,5],
                       height_ratios=[5,1]
                       )
    ax1 = pylab.subplot(gs[0])
    ax2 = pylab.subplot(gs[1])
    ax4 = pylab.subplot(gs[3])
    
    list_d = [[t[0] for t in row] for row in f]
    list_d = np.array(list_d)
    ax2.pcolor(list_d, cmap=pylab.get_cmap("Blues"))
    ax2.plot(path[:,1], path[:,0], c="C3")
    
    ax1.plot(A, range(len(A)))
    ax1.invert_xaxis()
    ax4.plot(B, c="C1")
    pylab.show()
    
    for line in path:
        pylab.plot(line, [A[line[0]], B[line[1]]], linewidth=0.2, c="gray")
    pylab.plot(A)
    pylab.plot(B)
    pylab.show()


a = [1.1, 0.1, 1.1, 1.1, 0.1, 2.1]
b = [0, 0, 0, 1, 0, 2]
c = [0, 0, 0, 0, 0, 0]

m = calc_dtw(a, b)

print(m)
print(m[-1][-1][0]/(len(a) + len(b)))
path = get_path(m)
plot_path(m, path, a, b)

m = calc_dtw(a, c)

print(m)
print(m[-1][-1][0]/(len(a) + len(c)))
path = get_path(m)
plot_path(m, path, a, c)