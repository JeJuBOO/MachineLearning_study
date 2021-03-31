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

plt.plot(s,hyperbolic_tan(s,a))
plt.grid(True)
plt.show()
