"""
このコードは行動分類用にプロット関係の関数をまとめたコードである
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_toolkits.mplot3d as mpl3d
import math
import pdb

class PlotMaker2D:
    _xmin = 0
    _xmax = 0
    _ymin = 0
    _ymax = 0

    def __init__(self, xmin, xmax, ymin, ymax, title="", xlabel="", ylabel="", figsize=(9.67, 9.67), dpi=100):
        plt.figure(dpi=dpi, figsize=figsize) # definition of size and resolution
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        return
    
    def adjust_margin(self, top=None, bottom=None, right=None, left=None):
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        return
    
    def set_lim(self, x_axis:bool, y_axis:bool):
        if (x_axis):
            plt.xlim(self._xmin, self._xmax)
        if (y_axis):
            plt.ylim(self._ymin, self._ymax)
        return
    
    def set_grid(self):
        plt.grid()
        return

    def create_line_graph(self, x, y, marker=None, color=None, linestyle=None):
        plt.plot(x,y,marker=marker, color=color, linestyle=linestyle)
        return

    def create_scatter_plot(self, x, y, marker=None, color=None, size=None, color_bar=False, color_map='Blues', h_p=None, v_p=None):
        if (h_p is not None):
            plt.hlines(h_p, self._xmin, self._xmax, linestyle='dashed', linewidth=0.5)
        if (v_p is not None):
            plt.vlines(v_p, self._ymin, self._ymax, linestyle='dashed', linewidth=0.5)

        plt.scatter(x, y, marker=marker, s=size, c=color, cmap=color_map)
        if (color_bar):
            plt.colorbar()
        return

    def create_bar_graph(self, left, height, width=0.8, color='blue', h_p=None):
        plt.xticks(rotation=90) # rotation
        if (h_p is not None):
            plt.hlines(h_p, self._xmin, self._xmax, linestyle='dashed', linewidth=1)
        plt.bar(left, height, width=width, color=color)
        return

    def show(self):
        plt.show()
        return

    def save_fig(self, filename):
        plt.savefig(filename)
        return

class PlotMaker3D:
    fig = None
    ax = None
    _xmin = 0
    _xmax = 0
    _ymin = 0
    _ymax = 0
    _zmin = 0
    _zmax = 0

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, title="", xlabel="", ylabel="", zlabel="", figsize=(19.2, 9.67), dpi=100):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_zlabel(zlabel)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax
        return

    def set_lim(self, x_axis:bool, y_axis:bool, z_axis:bool):
        if (x_axis):
            self.ax.set_xlim(self._xmin, self._xmax)
        if (y_axis):
            self.ax.set_ylim(self._ymin, self._ymax)
        if (z_axis):
            self.ax.set_zlim(self._zmin, self._zmax)
        return

    def create_scatter_plot(self, x, y, z, color="Blue"):
        """ 3次元散布図を作成する
        Parameter
            x, y, z : それぞれ各次元のリスト
            c   : 色名 (str) や16進数 (#00ff00), RGB (tuple([0, 1]の数値3次元)) での指定が可能（デフォルトは一色）"""
        # 散布図作成
        self.ax.scatter3D(x, y, z, c=color)
        return
    def show(self):
        plt.show()


if __name__ == "__main__":
    beta = np.array([8.24853498e+07, 1.11542649e+08, 3.72355777e+07])
    m = np.array([[0.14598122, 0.21648961], [0.40219225, 0.30129823], [0.30594259, 0.58127577]])
    nu = np.array([8.24853498e+07, 1.11542649e+08, 3.72355777e+07])
    W = np.array([[[1.86993649*(10**-6), -2.43005831*(10**-6)], [-2.43005831*(10**-6), 6.64427680*(10**-6)]],\
                    [[1.63108638*(10**-6), -7.02456611*(10**-7)], [-7.02456611*(10**-7), 1.72586475*(10**-6)]],\
                    [[5.04790981*(10**-6), -1.00129009*(10**-6)], [-1.00129009*(10**-6), 2.85946316*(10**-6)]]])

    # 事後分布から予測分布のパラメータを導出
    K = len(beta)
    D = len(m.T)
    mu = np.zeros((K, D))
    lam = np.zeros((K, D, D))
    nu_hat = np.zeros(K)
    for k in range(K):
        mu[k] = m[k]
        lam[k] = ((1 - D + nu[k]) * beta[k] / (1 + beta[k])) * W[k]
        nu_hat[k] = 1 - D + nu[k]

    # 分布の等高線を描く
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((K, N, N))
    for k in range(K):
        temp1 = np.exp(math.lgamma((nu_hat[k] + D) / 2) - math.lgamma(nu_hat[k] / 2))
        temp2 = np.linalg.det(lam[k]) ** (1 / 2) / ((np.pi * nu_hat[k]) ** (D / 2))
        for i in range(N):
            for j in range(N):
                x_vec = np.array([[x[i] - mu[k, 0]], [y[j] - mu[k, 1]]])
                temp3 = (1 + np.dot(x_vec.T, np.dot(lam[k], x_vec)) / nu_hat[k]) ** (-(nu_hat[k] + D) / 2)
                Z[k, i, j] = temp1 * temp2 * temp3
        plt.contour(X, Y, Z[k])
        plt.gca().set_aspect('equal')
    plt.show()
