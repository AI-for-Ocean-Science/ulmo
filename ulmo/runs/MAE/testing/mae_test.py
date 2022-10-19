import threading, time

time.sleep(30)

with open('test.txt', 'w') as f:
    f.write("hellow world. is everything going as planned?")
