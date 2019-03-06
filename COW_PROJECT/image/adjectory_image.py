#-*- encoding:utf-8 -*-
from PIL import Image
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
    
    def write(self):
        image_list = np.asarray(self.image) # 画像をarrayに変換
        plt.imshow(image_list) # 貼り付け
        plt.show()