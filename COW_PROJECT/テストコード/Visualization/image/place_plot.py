#-*- encoding:utf-8 -*-
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

class PlacePlotter:
    image:Image.Image
    __size = (633, 537) # (width, height)
    __top_left = (34.88328, 134.86187) # (latitude, longitude)
    __bottom_right = (34.88207, 134.86357) # (latitude, longitude)

    def __init__(self, back:bool):
        """ Parameter
                back	: bool  背景に衛星画像を表示するかどうか """
        if(back):
            self.image = Image.open("./テストコード/Visualization/image/background.jpg")
        else:
            self.image = Image.new('RGBA',self.__size,(255,255,255,255))
    
    def plot_places(self, pos_list, caption_list=None, color_list=None):
        """ リスト型の複数の位置情報を描画する
            Parameter
                pos_list    : list  (lat, lon) の2要素の2次元
                caption_list    : list  画像に表示するラベル (牛の個体番号など)
                color_list      : list  塗りつぶしの色のリスト """
        draw = ImageDraw.Draw(self.image) # 図形描画用オブジェクト
        for pos in pos_list:
            self.draw_circle(draw, 3, float(pos[0]), float(pos[1]), (0, 255, 0))
        image_list = np.asarray(self.image) # 画像をarrayに変換
        plt.imshow(image_list) # 貼り付け

    def draw_circle(self, draw, radius, latitude, longitude, color):
        """ 円を描く
            Parameter
                draw	:ImageDraw.Draw :図形描画用オブジェクト
                radius  : 円の半径
                latitude, longitude :緯度・経度
                color   : 色 """
        width = self.__bottom_right[1] - self.__top_left[1] # 正
        height = self.__bottom_right[0] - self.__top_left[0] # 負
        x = ((longitude - self.__top_left[1]) / width) * self.__size[0]
        y = ((latitude - self.__top_left[0]) / height) * self.__size[1]
        if((0 <= x and x <= self.__size[0]) and (0 <= y and y <= self.__size[1])):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
