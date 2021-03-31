import numpy as np
import matplotlib.pyplot as plt

#훈련집합
x=np.array([[0,0], [1,0], [0,1], [1,1]])
y=np.array([1,1,1,1])
#x_b = np.c_[np.ones((4,1)),x] #[bias x1 x2]

learning_rate = 0.2

theta = np.array([1,1,1], dtype=float)   #[w0 w1 w2]

def forward(x):
    return np.dot(x,theta[1:])+theta[0]

def predict(x):
    return np.where(forward(x)>0,1,-1) #계단 함수 목적함수가 0보다 크면1 아니면 -1

print("predict (before traning)",theta)

for epo in range(50):
    for x_val, y_val, in zip(x,y):
        update = learning_rate*(y_val-predict(x_val))
        theta[1:] += update*x_val
        theta[0] += update
print("predict (after traning)",theta)
