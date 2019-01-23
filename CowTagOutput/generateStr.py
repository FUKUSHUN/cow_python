#-*- coding: utf-8 -*-
import getCowTagList as cowtag
import datetime
import os
#CowDataFileListに登録する文字列を返す
class GenerateStr:
    dt:datetime
    cowId:cowtag.CowId
    def __init__(self, d):
        self.dt = d
        self.cowId = cowtag.CowId()

    #CowDataFileListに登録する文字列の生成
    def generateStr(self):
        gStr = "" #返り値
        #その日牛の入場があったかを調べる
        inList = self.cowId.getCowIn(self.dt)
        #その日機器の交換があったか調べる
        changeList = self.cowId.getDevChange(self.dt)
        if(len(inList) != 0):
            for newTag in inList:
                if(self.cowId.isCowExist(newTag[0], self.dt)):
                    gStr += "ida" + newTag[0] + ".setGPSID(" + newTag[1] + "); ida" + newTag[0] + ".setMobicolletID(" + newTag[2] + ");\n"
                else:
                    gStr += "CowTagID ida" + newTag[0] + "= new CowTagID(\"" + newTag[1] + "\", \"" + newTag[0] + "\", \"" + newTag[2] + "\");\n"
            gStr += "\n"
        if(len(changeList) != 0):
            for newDev in changeList:
                gStr += "ida" + newDev[0] + ".setGPSID(" + newDev[1] + "); ida" + newDev[0] + ".setMobicolletID(" + newDev[2] + ");\n"

        gStr += "//" + str(self.dt.month) + "/" + str(self.dt.day) +"\n"
        period = "null"
        if(0 < self.dt.day < 11):
            period = "上旬"
        elif(10 < self.dt.day < 21):
            period = "中旬"
        elif(20 < self.dt.day):
            period = "下旬"
                
        #基本情報
        date = str(self.dt.date()).replace("-", "")
        cowDataFile = "day" + date[4:]
        #ライン情報
        cowDataFileLine = "CowDataFile " + cowDataFile + " = new CowDataFile();\n"
        setGPSFileLine = cowDataFile + ".setGPSFilename(\"../GPSData/" + str(self.dt.year) + "年/" + str(self.dt.month) + "月" + period + "\");\n"
        setMobiFileLine = cowDataFile + ".setMobicolletFilename(\"../mobicolletData/" + str(date[2:4]) +"年" + str(self.dt.month) +"月/" + date + "\");\n"
        setCowIdLine = cowDataFile + ".setCowID(new ArrayList<CowTagID>(Arrays.asList(" + self.__changeCowListToStr() + ")));\n"
        filelistLine = "filelist.put(\"" + date + "\", "+ cowDataFile + ");\n"

        gStr += cowDataFileLine + setGPSFileLine + setMobiFileLine + setCowIdLine + filelistLine + "\n"

        return gStr

    #牛のリストを定型文にあうように変形して返す
    def __changeCowListToStr(self):
        cowList = self.cowId.getCowList(self.dt)
        cowListStr = ""
        for c in cowList:
            cowListStr += "ida" + str(c) + ","
        return cowListStr[:-1]
