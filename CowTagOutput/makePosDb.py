#-*- encoding:utf-8 -*-
import sqlite3
import pandas as pd
import re
import datetime
import makeTagDb

#日付とGPSIDからデータベースを作成する
class PosDb:
	parentPath = "" #IDA-GPSフォルダまで．この後にgpsId/date.txtが続く
	dbPath = "DB/PosDB/" #データベースの保管場所
	date:int #6桁の数字yymmdd
	cowId = 0
	gpsId = 0
	
	def __init__(self, cId, date):
		self.cowId = cId
		self.date = date
		f = open("Record/path.txt", "r", encoding='utf-8') #ここにそのパスが書いてあるのでローカルで設定すること
		self.parentPath = f.read()
		f.close()

	#データベースを作成する
	def makeDb(self):
		self.__getGpsId() #gpsIdを取得・格納
		df = self.__readCsv()
		conn = sqlite3.connect(self.dbPath + str(self.date)  + ".db")
		c = conn.cursor()
		c.execute("DROP TABLE IF EXISTS `" + str(self.cowId) + "`")
		c.execute("CREATE TABLE `"  + str(self.cowId) + "`(time CHAR(20), LATITUDE DOUBLE, LONGITUDE DOUBLE, VELOCITY DOUBLE)")
		for index, row in df.iterrows():
			if '$GPRMC' in index:
				time = row[0]
				lat = row[1]
				lon = row[2]
				vel = row[3]
				date = self.__transToYYMMDD(int(row[4]))
				dt = self.__transDateTime(time, date)
				sql = "INSERT INTO `" + str(self.cowId) + "`VALUES(" + "'" + dt.strftime("%Y/%m/%d %H:%M:%S") + "'" + "," + "'" + str(lat) + "'" + "," + "'" + str(lon) + "'" + "," + "'" + str(vel) + "'" + ")"
				c.execute(sql)
		conn.commit()
		itr = c.execute("SELECT * FROM `" + str(self.cowId) + "`")
		for row in itr:
			print(row)
		conn.close()
		
	#牛の個体番号から対応する牛のgpsIdを取得する
	def __getGpsId(self):
		#データベースの牛の個体番号のファイルを見る
		tagDB = makeTagDb.TagDb(str(self.cowId), self.date)
		#日付からgpsIdを取得する
		self.gpsId = tagDB.searchGpsId()
	
	#csvファイルから読み込む
	def __readCsv(self):
		#csv読み込み (見つからない場合のエラー回避)
		try:
			df = pd.read_table(filepath_or_buffer = self.parentPath + str(self.gpsId) + "/" + str(self.date) + ".txt", encoding = "utf-8", sep = ",", header = None, usecols = [0,1,3,5,7,9], names=('A', 'B', 'C', 'D', 'E','F'),index_col = 0, engine='python')
			df = df.fillna(0) #欠損等は0で補完
			return df
		except FileNotFoundError:
			df = pd.DataFrame([])
			return df
		
    #日付をdatetimeにして返す
	def __transDateTime(self,time, date):
		timeList = self.__devideEachDigit(time)
		hour = int(timeList[2])
		minu = int(timeList[1])
		sec = int(timeList[0])
		dateList = self.__devideEachDigit(date)
		year = int(2000 + dateList[2])
		month = int(dateList[1])
		day = int(dateList[0])
		dt = datetime.datetime(year, month, day, hour, minu, sec)
		return dt

	#6桁の数字を2桁ずつ取り出す
	def __devideEachDigit(self, num:int):
		array = []
		for i in range(3):
			try:
				array.append(num % 100)
			except TypeError:
				print(num)
			num = num // 100

		return array
        
	#ddmmyyをyymmddに変換する
	def __transToYYMMDD(self, date:int):
		array = self.__devideEachDigit(date)
		return 10000 * array[0] + 100 * array[1] + array[2]
		
	#指定されたデータベースのテーブルの一覧を表示する
	def show_all_table(self):
		conn = sqlite3.connect(self.dbPath + str(self.date)  + ".db")
		c = conn.cursor()
		sql = "SELECT name FROM sqlite_master WHERE type='table'" #SQLの文章 (テーブル一覧取得)
		itr = c.execute(sql) #SQL文の実行
		for x in itr:
			print(x)
		conn.close()
	
