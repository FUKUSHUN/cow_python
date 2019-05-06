#-*- encoding:utf-8 -*-
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

class Adjectory:
    image:Image.Image
    __size = (633, 537) # (width, height)
    __top_left = (34.88328, 134.86187) # (latitude, longitude)
    __bottom_right = (34.88207, 134.86357) # (latitude, longitude)

    """
	    back	:背景に衛星画像を表示するかどうか
    """
    def __init__(self, back:bool):
        if(back):
            self.image = Image.open("image/background.jpg")
        else:
            self.image = Image.new('RGBA',self.__size,(255,255,255,255))
    
    """
    移動の軌跡を描画する
    Parameter
        g_list: (lat, lon) の2要素の2次元リスト
    """
    def plot_moving_ad(self, g_list):
        draw = ImageDraw.Draw(self.image) # 図形描画用オブジェクト
        for pos in g_list:
            self.draw_circle(draw, pos[0], pos[1], 3)
            #print(pos[0], pos[1])
        image_list = np.asarray(self.image) # 画像をarrayに変換
        plt.imshow(image_list) # 貼り付け

    """
    休息の場所をプロットする
    Parameter
	    rest_g_list	:(lat, lon, time) の3要素の2次元リスト
    """
    def plot_rest_place(self, rest_g_list):
        draw = ImageDraw.Draw(self.image) # 図形描画用オブジェクト
        for pos in rest_g_list:
            if(pos[2] >= 1):
                self.draw_circle(draw, pos[0], pos[1], pos[2])
            #print(pos[0], pos[1], pos[2])
        image_list = np.asarray(self.image) # 画像をarrayに変換
        plt.imshow(image_list) # 貼り付け

    """
    Parameter
	    draw	:ImageDraw.Draw :図形描画用オブジェクト
        latitude, longitude :緯度・経度
        time    :そこにいた時間 (円の大きさに反映)
    """
    def draw_circle(self, draw, latitude, longitude, time):
        width = self.__bottom_right[1] - self.__top_left[1] #正
        height = self.__bottom_right[0] - self.__top_left[0] #負
        x = ((longitude - self.__top_left[1]) / width) * self.__size[0]
        y = ((latitude - self.__top_left[0]) / height) * self.__size[1]
        radius = 1 * time # 半径
        if((0 <= x and x <= self.__size[0]) and (0 <= y and y <= self.__size[1])):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
