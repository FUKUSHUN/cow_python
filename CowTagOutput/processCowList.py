#-*- coding: utf-8 -*-
import getCowTagList as cowtag
import generateStr as gst
import datetime
import os
#様々な処理をまとめて行う
class ProcessCowList:
	cowId:cowtag.CowId
	year = 0
	month = 0
	day = 0
	
	def __init__(self, cowId):
		self.cowId = cowId
		
	#指定された月の１日ごとの牛のリストを作成してファイル出力する
	def makeListFile(self):
		self.__setYMD(True, True, False)
		dt = datetime.datetime(self.year, self.month, 1)
		filename = "csv/" + str(dt.date())[:7] + ".csv"
		if(os.path.isfile(filename)):
			os.remove(filename)
		f = open(filename, "x", encoding = "utf-8")
		while(dt.month == self.month):
			rowStr = str(dt.strftime("%Y/%m/%d")) + ","
			cowList = self.cowId.getCowList(dt)
			for c in cowList:
				rowStr += str(c) + ","
			f.write(rowStr[:-1] + "\n")
			dt = dt + datetime.timedelta(days = 1)
		f.close()
		return

	#cowDataFileに登録する形式の文字列をファイルに出力する
	def makeFormatListFile(self):
		#開始日の入力
		self.__setYMD(True, True, True)
		dt = datetime.datetime(self.year, self.month, self.day)
		filename = "txt/" + str(dt.date())[:7] + ".txt"
		if(os.path.isfile(filename)):
			os.remove(filename)
		f = open(filename, "x", encoding = "utf-8")
		#次の月になるまで繰り返す
		while(dt.month < self.month + 1 and dt.year == self.year):
			gs = gst.GenerateStr(dt)
			formatList = gs.generateStr()
			f.write(formatList)
			dt = dt + datetime.timedelta(days = 1)
		f.close()	
		
	#IORecordの変更に伴ってCowTagLogを更新する
	def updateCowTagLog(self):
		self.__setYMD(True, True, False)
		self.cowId.updateLog(self.year, self.month)
		print(self.cowId.tagFilename + "が更新されました. 他のファイルは更新を行っていないので注意してください. " )
		
	#IORecordの追加に伴ってCowTagLogを追加する
	def addCowTagLog(self):
		self.__setYMD(True, True, False)
		self.cowId.addLog(self.year, self.month)
		print(self.cowId.tagFilename + "に追加されました. 他のファイルは更新を行っていないので注意してください. また，正しく追加されているか確認してください．" )
		
	#year, month, dayを登録する
	def __setYMD(self, y:bool, m:bool, d:bool):
		if(y):
			self.year = int(input("Year : "))
		if(m):
			self.month = int(input("Month (1 digit OK) : "))
		if(d):
			self.day = int(input("Day (1 digit OK) : "))
