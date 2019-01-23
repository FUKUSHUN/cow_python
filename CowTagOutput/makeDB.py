#-*- coding: utf-8 -*-
import getCowTagList as cowtag
import datetime
import makeTagDb
import makePosDb
import sys

def devideEachDigit(num):
    array = []
    for i in range(3):
        array.append(num % 100)
        num = num // 100

    return array


cowId = cowtag.CowId()
a = int(sys.argv[1])
b = int(sys.argv[2])
d = devideEachDigit(a)
dt = datetime.datetime(2000 + d[2], d[1], d[0])
d2 = devideEachDigit(b)
end = datetime.datetime(2000 + d2[2], d2[1], d2[0])
while dt <= end:
	cowList = cowId.getCowList(dt) #指定した日時に第一放牧場にいる牛に対して出力
	for cow in cowList:
		m = makePosDb.PosDb(int(cow), int(dt.strftime("%y%m%d")))
		m.makeDb()
	dt = dt + datetime.timedelta(days = 1)