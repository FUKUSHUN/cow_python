#-*- encoding:utf-8 -*-
import pandas as pd
import re
import datetime
import cows.cowshed as Cowshed
import time
import gc

def main():
	#dbFilepath = "../CowTagOutput/DB/PosDB/"
	start = datetime.datetime(2018, 2, 16, 0, 0, 0)
	end = datetime.datetime(2018, 2, 17, 0, 0, 0)

	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		start_time = time.time()
		cows = Cowshed.Cowshed(dt)
		end_time = time.time()
		dt = dt + datetime.timedelta(days = 1)
		print("{0}".format(end_time - start_time) + " [sec]")
		while(a < dt and a < end):
			start_time = time.time()
			df = cows.get_cow_list(a, a + datetime.timedelta(minutes = 60))
			end_time = time.time()
			print(a.strftime("%H:%M:%S") + " : {0}".format(end_time - start_time) + " [sec]")
			a = a + datetime.timedelta(minutes = 60)
			del df
			gc.collect()
		del cows
		gc.collect()
		a = dt

if __name__ == '__main__':
    main()

