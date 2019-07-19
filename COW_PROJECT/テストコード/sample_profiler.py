#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import pandas as pd
import re
import datetime
import cow.cowshed as Cowshed
import time
import gc
from memory_profiler import profile

@profile
def main():
	dbFilepath = "../CowTagOutput/DB/PosDB/"
	start = datetime.datetime(2018, 2, 20, 9, 0, 0)
	end = datetime.datetime(2018, 2, 21, 1, 0, 0)

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
			df = cows.get_cow_list(a, a + datetime.timedelta(minutes = 10))
			end_time = time.time()
			print(a.strftime("%H:%M:%S") + " : {0}".format(end_time - start_time) + " [sec]")
			a = a + datetime.timedelta(minutes = 10)
			del df
			gc.collect()
		del cows
		gc.collect()
		a = dt

if __name__ == '__main__':
    main()

