#-*- encoding:utf-8 -*-
import time
import datetime

start = datetime.datetime(2018, 2, 20, 22, 0, 0)
end = datetime.datetime(2018, 2, 20, 23, 0, 0)

a = start

start_time = time.time()
while(start <= a and a < end):
	a = a + datetime.timedelta(minutes = 1)
end_time = time.time()
print("{0}".format(end_time - start_time) + " [sec]")

start = start + datetime.timedelta(hours = 1)
end = end + datetime.timedelta(hours = 1)
end = end - datetime.timedelta(seconds = 1)
print(end.strftime("%H:%M:%S"))

a = start
start_time = time.time()
while(start <= a and a < end):
	a = a + datetime.timedelta(minutes = 1)
end_time = time.time()
print("{0}".format(end_time - start_time) + " [sec]")
