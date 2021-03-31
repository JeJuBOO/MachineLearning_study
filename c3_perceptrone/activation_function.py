import numpy as np
import matplotlib.pyplot as plt

s = np.arange(-10,10,0.1)
a = 1

def stairs_function(s):
    return np.where(s>=0,1,-1)

def logistic_sig(s,a):
    return 1/(1+np.exp(-a*s))

def hyperbolic_tan(s,a):
    return 2/(1+np.exp(-a*s))-1

def softflus(s):
    return np.log(1+np.exp(s))

def ReLU(s):
    return np.maximum(0,s)

def leaky_ReLU(s):
    return np.maximum(0.01*s,s)

plt.plot(s,leaky_ReLU(s))
plt.grid(True)
plt.show()
