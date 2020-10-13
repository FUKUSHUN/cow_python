"""
このコードは行動分類用にプロット関係の関数をまとめたコードである
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpl_toolkits.mplot3d as mpl3d

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