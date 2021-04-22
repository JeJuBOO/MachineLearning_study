import numpy as np
import matplotlib.pyplot as plt
import time
import random as rd
from sklearn.datasets import load_iris

iris = load_iris()
# 훈련집합
x = iris.data[:100,[0,2]]
Y = iris.target[:100]
X = np.c_[np.ones((len(x),1)),x]

# 파라미터
in_node_n = X[1].size
learning_rate = 0.2
hidden_node_n = 4
out_node_n = 1

def onehot(y,n):
    return np.insert(np.zeros((1,n-1)),y,1)

# 목적함수
def mse(y,o):
    return (1/2)*np.sum((y-o)**2)
def cee(y,o):
    return -np.sum(o*np.log(y+1e-7))


# 활성화 함수

def stairs_function(s):
    return np.where(s>=0,1,-1)

def logistic_sig(s,a=1):
    return 1/(1+np.exp(-a*s))

def d_logistic_sig(s,a=1):
    return a*logistic_sig(s,a)*(1-logistic_sig(s,a))

def ReLU(s):
    return np.maximum(0,s)

plot_x = np.arange(np.min(x[:,:1])-1,np.max(x[:,:1])+1,0.1)


U1 = np.random.random((in_node_n,hidden_node_n))
U2 = np.random.random((hidden_node_n+1,out_node_n))

dU1 = np.zeros((in_node_n,hidden_node_n))
dU2 = np.zeros((hidden_node_n+1,out_node_n))
start_t = time.time()
epo = 0
while True:
    epo+=1

    # 미니배치
    rand_index = rd.sample(range(len(X)),rd.randint(1,len(X)))
    X_mini = X[rand_index,:]
    Y_mini = Y[rand_index]

    dU1 = np.zeros((in_node_n,hidden_node_n))
    dU2 = np.zeros((hidden_node_n+1,out_node_n))
    
    i=0
    error = np.zeros(len(rand_index))
    for x_val, y_val, in zip(X_mini,Y_mini):
        # 전방 계산
        hidden_node = np.insert(logistic_sig(x_val.dot(U1)),0,1)
        out_node = logistic_sig(hidden_node.dot(U2))

        # 오류역전파
        gradient = (y_val-out_node) * d_logistic_sig(hidden_node.dot(U2))
        dU2 += -gradient*hidden_node.reshape(-1,1)
        gradient2 = gradient.dot(U2[1:].T) * d_logistic_sig(x_val.dot(U1))
        dU1 += -gradient2*x_val.reshape(-1,1)

        error[i] = mse(y_val,out_node)
        i+=1
    # 가중치 갱신
    U2 += -learning_rate*dU2/len(rand_index)
    U1 += -learning_rate*dU1/len(rand_index)

    # 목적함수를 이용한 오차 확인
    if np.sum(error)/len(rand_index) <= 1e-5 and epo > 1000:
        t = time.time()-start_t
        print(y_val,out_node,error,len(rand_index))
        print(mse(y_val,out_node))
        break

# 테스트
i=0
error=0
out_node=np.zeros(len(Y))
for x_val, y_val, in zip(X,Y):
    hidden_node = np.insert(logistic_sig(x_val.dot(U1)),0,1)
    out_node = logistic_sig(hidden_node.dot(U2))
    print(out_node)
    error += mse(y_val,out_node)
    i += 1

# U1의 특징공간
# for i in range(hidden_node_n):
#     plt.plot(plot_x,-U1[0,i]/U1[2,i]-U1[1,i]*plot_x/U1[2,i])
# plt.plot(x[:,0],x[:,1],marker='o', linestyle = 'None')
# plt.grid()
# print(out_node[:50])
# print(out_node[50:100])
# print(out_node[100:])
# 평균 수렴 시간
print("mean time : ",t)
print("mean error : ",error/len(Y))

plt.show()
