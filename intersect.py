#!/usr/bin/python

class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


a = Point(0,0)
b = Point(0,1)
c = Point(1,1)
d = Point(1,0)


print intersect(a,b,c,d)
print intersect(a,c,b,d)
print intersect(a,d,b,c)