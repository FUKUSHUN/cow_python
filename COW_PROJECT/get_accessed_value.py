#-*- encoding:utf-8 -*-
import pandas as pd
import cows.cows_community as comm
import cows.cowshed as Cowshed
import datetime
import gc

def main():
	start = datetime.datetime(2018, 2, 16, 0, 0, 0)
	end = datetime.datetime(2018, 2, 16, 1, 0, 0)

	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		cows = Cowshed.Cowshed(dt)
		dt = dt + datetime.timedelta(days = 1)
		while(a < dt and a < end):
            df = cows.get_cow_list(a, a + datetime.timedelta(minutes = 10))
            a = a + datetime.timedelta(minutes = 10)
			del df
			gc.collect()
		del cows
		gc.collect()
		a = dt

if __name__ == '__main__':
    main()