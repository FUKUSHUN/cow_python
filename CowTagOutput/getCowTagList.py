#-*- coding: utf-8 -*-
import datetime
import csv
import re
import os
import sys
#月初ごとにログファイルとしてとっておき，そこからの牛の入出を調べることで該当日の牛のリストを取得する
class CowId:
	tagFilename = "Record/CowTagLog.csv" #月初ごとの牛のリストをまとめたファイル名
	recordFilename = "Record/IORecord.csv" #入退出変更をまとめたファイル名
	existFilename = "Record/ExistRecord.csv" #一度第1放牧場に出た牛が第2放牧場や分娩房も含めてまだ存在しているかをまとめたファイル
	pastExistFilename = "Record/PastExistRecord.csv" #過去に存在した牛全ての個体番号をまとめたファイル
	iniDt = datetime.datetime(2018, 2, 1) #一番古いデータ(これ以上は戻れない)
		
	#月初に記録するログファイルを更新する
	def updateLog(self, year, month):
		d1 = datetime.datetime(year, month, 1)
		if(self.iniDt < d1):
			d2 = d1 - datetime.timedelta(days = 1)
			self.updateLog(d2.year, d2.month) #再帰
			self.addLog(d1.year, d1.month)
			return
		else:
			#CowTagFile
			f = open(self.tagFilename, "r")
			rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
			header = next(rd) #全てのファイルの元祖のメンバー
			f.close()
			if(os.path.isfile(self.tagFilename)):
				os.remove(self.tagFilename)
			f = open(self.tagFilename, "x", encoding = "utf-8")
			iniRow = ""
			for item in header:
				iniRow += item + ","
			f.write(iniRow[:-1])
			f.close
			#ExistFile
			f = open(self.existFilename, "r")
			rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
			header = next(rd)
			f.close()
			if(os.path.isfile(self.existFilename)):
				os.remove(self.existFilename) #一旦消去
			f = open(self.existFilename, "x", encoding = "utf-8")
			iniRow = ""
			for item in header:
				iniRow += item + ","
			f.write(iniRow[:-1])
			f.close
			#PastExistFile
			f = open(self.pastExistFilename, "r")
			rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
			header = next(rd)
			f.close()
			if(os.path.isfile(self.pastExistFilename)):
				os.remove(self.pastExistFilename) #一旦消去
			f = open(self.pastExistFilename, "x", encoding = "utf-8")
			iniRow = ""
			for item in header:
				iniRow += item + ","
			f.write(iniRow[:-1])
			f.close
			return
		
	#ログを追加する
	def addLog(self, year, month):
		d = datetime.datetime(year, month, 1)
		#CowTagFile
		f = open(self.tagFilename, "a", encoding = "utf-8")
		cowList = self.__getCowListFirst(d)
		f.write("\n")
		rowStr = str(d.strftime("%Y/%m/%d")) + ","
		for c in cowList:
			rowStr += str(c) + ","
		f.write(rowStr[:-1])
		f.close()
		#ExistFile
		f = open(self.existFilename, "a", encoding = "utf-8")
		cowList = self.__getExistListFirst(d)
		f.write("\n")
		rowStr = str(d.strftime("%Y/%m/%d")) + ","
		for c in cowList:
			rowStr += str(c) + ","
		f.write(rowStr[:-1])
		f.close()
		#PastExistFile
		f = open(self.pastExistFilename, "a", encoding = "utf-8")
		cowList = self.__getPastExistListFirst(d)
		f.write("\n")
		rowStr = str(d.strftime("%Y/%m/%d")) + ","
		for c in cowList:
			rowStr += str(c) + ","
		f.write(rowStr[:-1])
		f.close()
		return
		
	#指定日に牛の入場があった場合，その情報を返す
	def getCowIn(self, dt):
		#牛の入出情報を取得する
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		inList = []
		cowList = []
		if(dt != self.iniDt): #というかself.iniDtの時だけ特別に空にしている
			dtb = dt - datetime.timedelta(days = 1)
			cowList = self.getCowList(dtb)
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(dt == d):
				io = row[1]
				cowId = row[2]
				#gpsId = row[3], mobiId = row[4]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						inList.append(row[2:])
			if(dt < d):
				f.close()
				return inList
		f.close()
		return inList
            
    #指定日に牛の退場があった場合，その情報を返す
	def getCowOut(self, dt):
		#牛の退出情報を取得する
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		outList = []
		cowList = []
		if(dt != self.iniDt): #というかself.iniDtの時だけ特別に空にしている
			dtb = dt - datetime.timedelta(days = 1)
			cowList = self.getCowList(dtb)
		for row in rd:
 			d = datetime.datetime.strptime(row[0], "%m/%d/%Y")
 			if(dt == d):
 				io = row[1]
 				cowId = row[2]
 				#gpsId = row[3], mobiId = row[4]
 				if(re.match("OUT", io)):
 					if(cowId in cowList):
 						outList.append(row[2:])
 			if(dt < d):
 				f.close()
 				return outList
		f.close()
		return outList

	#指定日に機器の交換があった場合，その情報を返す
	def getDevChange(self, dt):
		#牛の入出情報を取得する
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		changeList = []
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(dt == d):
				io = row[1]
				cowId = row[2]
				#gpsId = row[3], mobiId = row[4]
				if(re.match("CHANGE", io)):
					changeList.append(row[2:])
			if(dt < d):
				f.close()
				return changeList
		f.close()
		return changeList
		
	#指定日に牛の存在リストに牛が存在しているかを調べる
	def isCowExist(self, cowId, dt):
		dtb = dt - datetime.timedelta(days = 1)
		cowList = self.getExistList(dtb)
		if(cowId in cowList):
			return True
		else:
			return False
	#指定した日の牛のリストを取得する (目的：今第一放牧場に存在する牛を知りたい)
	def getCowList(self, dt):
		if(dt.day == 1):
			return self.__getCowListFirst(dt)
		else:
			return self.__getCowListSecond(dt)
			
	#指定した日の牛の存在リストを取得する (目的：今存在する牛を知りたい)
	def getExistList(self, dt):
		if(dt.day == 1):
			return self.__getExistListFirst(dt)
		else:
			return self.__getExistListSecond(dt)
			
	#指定した日の牛の存在リストを取得する (目的：過去存在する全ての牛を知りたい)
	def getPastExistList(self, dt):
		if(dt.day == 1):
			return self.__getPastExistListFirst(dt)
		else:
			return self.__getPastExistListSecond(dt)

	##指定した日(各月の1日以外)の牛のリストを取得する (目的：今第一放牧場に存在する牛を知りたい)
	def __getCowListSecond(self, dt):
		cowList = self.__getCowListBegin(dt)
		#牛の入出情報を取得する
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		d1 = datetime.datetime(dt.year, dt.month, 1)
		for row in rd:
			d2 = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(d1 <= d2 and d2 <= dt):
				io = row[1]
				cowId = row[2]
				gpsId = row[3]
				mobiId = row[4]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						cowList.append(cowId)
						cowList.sort()
				elif(re.match("OUT", io)):
					if(cowId in cowList):
						cowList.remove(cowId)
					else:
						print(str(cowId) + "は" + str(d2.month) + "月リストに含まれません. ")
			if(dt < d2):
				break
		f.close()
		return cowList
		
	##各月1日だけ別の処理が必要となる (目的：今第一放牧場に存在する牛を知りたい)
	def __getCowListFirst(self, dt):
		if(dt == self.iniDt):
			cowList = self.__getCowListBegin(dt)
		else:
			dtb = dt - datetime.timedelta(days = 1)
			cowList = self.__getCowListSecond(dtb)
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(d == dt):
				io = row[1]
				cowId = row[2]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						cowList.append(cowId)
						cowList.sort()
				elif(re.match("OUT", io)):
					if(cowId in cowList):
						cowList.remove(cowId)
					else:
						print(str(cowId) + "は" + str(dt.month) + "月リストに含まれません. ")
			if(dt < d):
				break
		f.close()
		return cowList
	
	##指定した日(各月の1日以外)の牛の存在リストを取得する (目的：今存在する牛を知りたい)
	def __getExistListSecond(self, dt):
		cowList = self.__getExistListBegin(dt)
		#牛の入出情報を取得する
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		d1 = datetime.datetime(dt.year, dt.month, 1)
		for row in rd:
			d2 = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(d1 <= d2 and d2 <= dt):
				io = row[1]
				cowId = row[2]
				#gpsId = row[3], mobiId = row[4]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						cowList.append(cowId)
						cowList.sort()
				elif(re.match("OUT", io)):
					isExist = row[5]
					if(re.match("NOT", isExist) and (cowId in cowList)):
						cowList.remove(cowId)
			if(dt < d2):
				break
		f.close()
		return cowList
	
	##指定した日(各月の1日)の牛の存在リストを取得する (目的：今存在する牛を知りたい)
	def __getExistListFirst(self, dt):
		if(dt == self.iniDt):
			cowList = self.__getExistListBegin(dt)
		else:
			dtb = dt - datetime.timedelta(days = 1)
			cowList = self.__getExistListSecond(dtb)
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(d == dt):
				io = row[1]
				cowId = row[2]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						cowList.append(cowId)
						cowList.sort()
				elif(re.match("OUT", io)):
					isExist = row[5]
					if((cowId in cowList) and re.match("NOT", isExist)):
						cowList.remove(cowId)
					else:
						print(str(cowId) + "は" + str(dt.month) + "月リストに含まれません. ")
			if(dt < d):
				break
		f.close()
		return cowList

	##指定した日(各月の1日以外)の牛の存在リストを取得する (目的：過去存在する全ての牛を知りたい)
	def __getPastExistListSecond(self, dt):
		cowList = self.__getPastExistListBegin(dt)
		#牛の入出情報を取得する
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		d1 = datetime.datetime(dt.year, dt.month, 1)
		for row in rd:
			d2 = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(d1 <= d2 and d2 <= dt):
				io = row[1]
				cowId = row[2]
				#gpsId = row[3], mobiId = row[4]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						cowList.append(cowId)
						cowList.sort()
			if(dt < d2):
				break
		f.close()
		return cowList

	##指定した日(各月の1日)の牛の存在リストを取得する (目的：過去存在する全ての牛を知りたい)
	def __getPastExistListFirst(self, dt):
		if(dt == self.iniDt):
			cowList = self.__getPastExistListBegin(dt)
		else:
			dtb = dt - datetime.timedelta(days = 1)
			cowList = self.__getPastExistListSecond(dtb)
		f = open(self.recordFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		header = next(rd)  # ヘッダーを読み飛ばしたい時
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%m/%d/%Y")
			if(d == dt):
				io = row[1]
				cowId = row[2]
				if(re.match("IN", io)):
					if(cowId not in cowList):
						cowList.append(cowId)
						cowList.sort()
			if(dt < d):
				break
		f.close()
		return cowList

	###月初の牛のリストを取得する (目的：今第一放牧場に存在する牛を知りたい)
	def __getCowListBegin(self, dt):
		f = open(self.tagFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		cowList = []
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%Y/%m/%d")
			if(d.year == dt.year and d.month == dt.month):
				cowList = row[1:]
				break
		f.close()
		if(len(cowList) == 0):
			print("牛のリストがありません．ファイルを確認してください．" + str(dt.date()))
			sys.exit()
		else:
			return cowList
			
	###月初の牛の存在リストを取得する (目的：今存在する牛を知りたい)
	def __getExistListBegin(self, dt):
		f = open(self.existFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		cowList = []
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%Y/%m/%d")
			if(d.year == dt.year and d.month == dt.month):
				cowList = row[1:]
				break
		f.close()
		if(len(cowList) == 0):
			print("牛のリストがありません．ファイルを確認してください．" + str(dt.date()))
			sys.exit()
		else:
			return cowList

	###月初の牛の存在リストを取得する (目的：過去存在する全ての牛を知りたい)
	def __getPastExistListBegin(self, dt):
		f = open(self.pastExistFilename, "r")
		rd = csv.reader(f, delimiter = ",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
		cowList = []
		for row in rd:
			d = datetime.datetime.strptime(row[0], "%Y/%m/%d")
			if(d.year == dt.year and d.month == dt.month):
				cowList = row[1:]
				break
		f.close()
		if(len(cowList) == 0):
			print("牛のリストがありません．ファイルを確認してください．" + str(dt.date()))
			sys.exit()
		else:
			return cowList
			
	def getIniDt(self):
		return self.iniDt

