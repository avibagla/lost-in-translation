import random
import time
from numpy import *
def reset():
	m = 2000
	n = 3000
	A = [[(i+1)*1.]*m for i in range(n)]
	x = [sum([e[i] for e in A]) for i in range(m)]
	return A, x

A, x = reset()
start = time.clock()
for i in range(len(A)): 
	for j in range(len(A[i])): 
		A[i][j] /= x[j]
print time.clock() - start		

A, x = reset()
start = time.clock()
for i in xrange(len(A)): 
	for j in xrange(len(A[i])): 
		A[i][j] /= x[j]
print time.clock() - start		

A,x = reset()
start = time.clock()
A = [[e[j]/x[j] for j in xrange(len(e))] for e in A]
print time.clock() - start		

A,x = reset()
start = time.clock()
A = [map(lambda t, total: t/total, e,x) for e in A]
print time.clock() - start

A = arange(2000*3000).reshape(2000,3000).astype(float64)
x = A.sum(axis = 0)
start = time.clock()
A = A / x[newaxis,:]
print time.clock() - start
