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

def d_stairs_function(s): #s=0 불가, 이외는 0 이다.
    return np.where(s=0,null,0)

def d_logistic_sig(s,a):
    return a*logistic_sig(s,a)*(1-logistic_sig(s,a))

def d_hyperbolic_tan(s,a):
    return 0.5*(1-hyperbolic_tan(s,a)^2)

def d_softflus(s):
    return 1/(1+np.exp(-s))

def ReLU(s):
    return np.where(s>=0,1,0)

def leaky_ReLU(s):
    return np.where(s>=0,1,0.01)
