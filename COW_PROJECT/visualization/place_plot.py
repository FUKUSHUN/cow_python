#-*- encoding:utf-8 -*-
import numpy as np
import os,sys
import datetime
import pandas as pd
import cv2.cv2 as cv2
import pdb

# 自作クラス
import visualization.place_plot as place_plot

class PlacePlotter:
    """ 1枚の画像を作成するクラス, imageはnp.ndarray, save_imageで画像を保存 """
    image:np.ndarray
    __size = (633, 557) # (width, height)
    __top_left = (34.88328, 134.86187) # (latitude, longitude)
    __bottom_right = (34.88207, 134.86357) # (latitude, longitude)

    def __init__(self, back:bool):
        """ Parameter
                back	: bool  背景に衛星画像を表示するかどうか """
        if(back):
            self.image = cv2.imread("./visualization/image/background.jpg")
        else:
            self.image = cv2.imread('RGBA',self.__size,(255,255,255,255))
    
    def plot_places(self, pos_list, caption_list=None, color_list=None, label=""):
        """ リスト型の複数の位置情報を描画する
            Parameter
                pos_list    : list  (lat, lon) の2要素の2次元
                caption_list    : list  画像に表示するラベル (牛の個体番号など)
                color_list      : list  塗りつぶしの色のリスト """
        for i, pos in enumerate(pos_list):
            x, y = self._draw_circle(3, float(pos[0]), float(pos[1]), (0, 255, 0))
            if (caption_list is not None):
                cap = caption_list[i]
                cv2.putText(self.image, str(cap), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(self.image, label, (5, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        return

    def _draw_circle(self, radius, latitude, longitude, color):
        """ 円を描く
            Parameter
                draw	:ImageDraw.Draw :図形描画用オブジェクト
                radius  : 円の半径
                latitude, longitude :緯度・経度
                color   : 色 """
        width = self.__bottom_right[1] - self.__top_left[1] # 正
        height = self.__bottom_right[0] - self.__top_left[0] # 負
        x = int(((longitude - self.__top_left[1]) / width) * self.__size[0])
        y = int(((latitude - self.__top_left[0]) / height) * self.__size[1])
        if((0 <= x and x <= self.__size[0]) and (0 <= y and y <= self.__size[1])):
            cv2.circle(self.image, (x, y), radius, color)
        return x, y
    
    def get_image(self):
        return self.image

    def save_image(self, filename):
        cv2.imwrite(filename, self.image)
        return


class PlotMaker:
    """ 複数枚の画像から位置プロット動画を作成するクラス """
    video_filename: str
    width: int
    height: int

    def __init__(self):
        self.video_filename = "./visualization/movie/"
        size = cv2.imread("./visualization/image/background.jpg").shape
        self.height = size[0]
        self.width = size[1]

    def make_movie(self, df:pd.DataFrame):
        """ 動画を作成する """
        # 画像の系列の取得
        images = self._make_sequence_images(df)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video  = cv2.VideoWriter(self.video_filename, fourcc, 10.0, (self.width, self.height))
        # 画像を繋げて動画にする
        for img in images:
            video.write(img)
        # 出力
        video.release()

    def _make_sequence_images(self, df):
        """ 指定時刻の画像データを系列として保持する """
        images = []
        start = df.index[0]
        end = df.index[len(df.index)-1]
        self.video_filename += start.strftime("%Y%m%d/")
        self._confirm_dir(self.video_filename) # ディレクトリを作成
        self.video_filename += start.strftime("%H%M%S-") + end.strftime("%H%M%S.mp4") # ファイル名を決定
        for time, data in df.iterrows():
            image = self._make_image(data, time)
            images.append(image)
        return images

    def _make_image(self, df:pd.DataFrame, time:datetime.datetime):
        """ 1時刻から画像を作成する """
        plotter = PlacePlotter(True)
        caption_list = []
        pos_list = []
        for cow_id, data in df.iteritems():
            pos = (data[0], data[1])
            caption_list.append(cow_id)
            pos_list.append(pos)
        plotter.plot_places(pos_list, caption_list=caption_list, label=time.strftime("%Y/%m/%d %H:%M:%S"))
        image = plotter.get_image()
        return image

    def _confirm_dir(self, dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return