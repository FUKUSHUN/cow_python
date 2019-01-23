#-*- coding: utf-8 -*-
import getCowTagList as cowtag
import datetime
import sqlite3
import os
#各牛ごとに日付を主キーにして牛の個体番号とGPSID・MobicolletIDのデータベースを作成する
class TagDb:
    dbPath = "DB/TagDB/" #データベースの保管場所
    cowNum:str #牛の個体番号
    date:int #6桁yymmdd (int)

    def __init__(self, cowNum, date):
        self.cowNum = cowNum
        self.date = date
     
    #牛の個体番号からGPSIDの検索をかける(日付はdateより得る)   
    def searchGpsId(self):
        cowId = cowtag.CowId()
        dt = self.__transDateTime(self.date) #この日のGPSIdを検索
        conn = sqlite3.connect(self.dbPath + self.cowNum + ".db") #DB接続
        c = conn.cursor()
        gId = c.execute("SELECT GPSId FROM TagInfo WHERE time = " + "'" + dt.strftime("%Y/%m/%d") + "'") #int
        gpsId = 0
        for row in gId:
            gpsId = row[0]
        conn.close()
        return gpsId
        
    #各日付のGPSIdとMobicolletIdのデータベースを作成して出力する
    def makeDb(self):
        cowId = cowtag.CowId()
        conn = sqlite3.connect(self.dbPath + self.cowNum + ".db")
        cu = conn.cursor()
        cu.execute("DROP TABLE IF EXISTS TagInfo")
        cu.execute("CREATE TABLE TagInfo(time CHAR(20), GPSId INT, MobiId INT)")
        dt = cowId.getIniDt()# + datetime.timedelta(days = 1) #この日から探し出す
        end = self.__transDateTime(self.date) #この日まで
        isInPasture = False
        while(dt <= end):
            inList = cowId.getCowIn(dt)
            for c in inList:
                if(self.cowNum == c[0]):
                    gpsId = c[1]
                    mobiId = c[2]
                    isInPasture = True
            changeList = cowId.getDevChange(dt)
            for c in changeList:
                if(self.cowNum == c[0]):
                    gpsId = c[1]
                    mobiId = c[2]
                    isInPasture = True
            outList = cowId.getCowOut(dt)
            for c in outList:
                if(self.cowNum == c[0]):
                    isInPasture = False
            if(isInPasture):
                sql = "INSERT INTO TagInfo VALUES(" + "'" + dt.strftime("%Y/%m/%d") + "'" + "," + "'" + str(gpsId) + "'" + "," + "'" + str(mobiId) + "')"
                cu.execute(sql)
            dt = dt + datetime.timedelta(days = 1)
        conn.commit()
        itr = cu.execute("SELECT * FROM TagInfo")
        for row in itr:
            print(row)
        conn.close()
        return

    #日付をdatetimeにして返す
    def __transDateTime(self, date):
    	dateList = self.__devideEachDigit(date)
    	year = int(2000 + dateList[2])
    	month = int(dateList[1])
    	day = int(dateList[0])
    	dt = datetime.datetime(year, month, day)
    	return dt

    #6桁の数字を2桁ずつ取り出す
    def __devideEachDigit(self, num):
    	array = []
    	for i in range(3):
        	array.append(num % 100)
        	num = num // 100

    	return array


