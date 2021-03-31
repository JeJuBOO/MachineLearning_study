import numpy as np
import scipy.linalg as linalg

A = np.array([[1,2,-1],[2,7,4],[0,4,-1]])
b = np.array([1,0,1.2])

# matrix-vector multiplication
y1 = np.matmul(A,b)
y2 = np.dot(A,b)
print(y1)
