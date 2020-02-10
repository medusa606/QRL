def adderScope(a):
	a = a + 1
	return a

a = 3
for i in range(0,5):
	print("before", a)
	a = adderScope(a)
	print("after", a)