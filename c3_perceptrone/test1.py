import numpy as np
from sklearn.datasets import load_iris
import random as rd
# iris = load_iris()
# x = iris.data[:100,[0,2]]
# Y = iris.target[:100]
# X = np.c_[np.ones((len(x),1)),x]
# rand_index = rd.sample(range(len(X)),rd.randint(1,len(X)))
# X_mini = X[rand_index,:]
# Y_mini = Y[rand_index]
# print(X_mini)
# def mse(y,o):
#     return 0.5*np.sum((y-o)**2)
# def cee(y,o):
#     return -np.sum(o*np.log(y+1e-7))
# def onehot(y,n):
#     return np.insert(np.zeros((1,n-1)),y,1)
#
# print(mse(onehot(2,3),[0, 0, 1]))

print(np.insert(np.zeros((1,1)),1,1))
