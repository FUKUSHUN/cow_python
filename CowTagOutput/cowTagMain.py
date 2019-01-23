#-*- coding: utf-8 -*-
import processCowList as pcl
import getCowTagList as cowtag
import generateStr as gst
import datetime
import os

cowId = cowtag.CowId()
p = pcl.ProcessCowList(cowId)

print("指定月の牛のリストのcsvを作成する ---> 1")
print("指定日からその月の牛の入力txtを作成する ---> 2")
print("IORecordの変更に伴ってCowTagLogを更新する ---> 3")
print("IORecordの追加に伴ってCowTagLogを追加する ---> 4")
choice = int(input("何を行いますか"))
if(choice == 1):
    p.makeListFile()
elif(choice == 2):
    p.makeFormatListFile()
elif(choice == 3):
    p.updateCowTagLog()
elif(choice == 4):
    p.addCowTagLog()
else:
    print("不正な入力です")

