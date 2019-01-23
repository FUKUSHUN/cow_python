#-*- coding: utf-8 -*-
import getCowTagList as cowtag
import datetime
import makeTagDb
import makePosDb

def devideEachDigit(num):
    array = []
    for i in range(3):
        array.append(num % 100)
        num = num // 100

    return array
    
cowId = cowtag.CowId()

print("TagDBを編集 ---> 1")
print("PosDBを編集 ---> 2")
print("Tag検索 ---> 3")
print("Table一覧 ---> 4")
choice = int(input("何を行いますか"))
if(choice == 1):
	date = int(input("yymmdd : "))
	d = devideEachDigit(date)
	dt = datetime.datetime(2000 + d[2], d[1], d[0])
	cowList = cowId.getPastExistList(dt) #今までに第一放牧場にいた全ての牛に対して出力
	for cow in cowList:
		m = makeTagDb.TagDb(cow, date)
		m.makeDb()
    	
elif(choice == 2):
	date = int(input("yymmdd : "))
	d = devideEachDigit(date)
	dt = datetime.datetime(2000 + d[2], d[1], d[0])
	cowList = cowId.getCowList(dt) #指定した日時に第一放牧場にいる牛に対して出力
	for cow in cowList:
		m = makePosDb.PosDb(int(cow), date)
		m.makeDb()

elif(choice == 3):
	date = int(input("yymmdd : "))
	d = devideEachDigit(date)
	dt = datetime.datetime(2000 + d[2], d[1], d[0])
	cowList = cowId.getCowList(dt) #指定した日時に第一放牧場にいる牛に対して出力
	for cow in cowList:
		m = makeTagDb.TagDb(cow, date)
		a = m.searchGpsId()
		if a == 0:
			print(cow + " : GPS未装着")
		else:
			print(cow + " : " + str(a))

elif(choice == 4):
	date = int(input("yymmdd : "))
	d = devideEachDigit(date)
	dt = datetime.datetime(2000 + d[2], d[1], d[0])
	m = makePosDb.PosDb(0, date) #cowIdが必要ないため0としてオブジェクトを作成している
	m.show_all_table()

else:
    print("不正な入力です")

