import numpy as np
import matplotlib.pyplot as plt

# 훈련집합
x = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([[0],[1],[1],[0]])
X = np.c_[np.ones((4,1)),x]

# 파라미터
in_node_n = X[1].size
learning_rate = 0.2
hidden_node_n = 3
out_node_n = Y[1].size

# 가중치 행렬
U1 = np.random.random((in_node_n,hidden_node_n)) #3*2
U2 = np.random.random((hidden_node_n+1,out_node_n)) #3*1

# 목적함수
def mse(y,o):
    return 0.5*np.sum((y-o)**2)
def cee(y,o):
    return -np.sum(o*np.log(y+1e-7))


# 활성화 함수
def logistic_sig(s,a=1):
    return 1/(1+np.exp(-a*s))

def d_logistic_sig(s,a=1):
    return a*logistic_sig(s,a)*(1-logistic_sig(s,a))

for epo in range(100000):
    for x_val, y_val, in zip(X,Y):
        # 전방 계산
        hidden_node = np.insert(logistic_sig(x_val.dot(U1)),0,1) # 1*3
        out_node = logistic_sig(hidden_node.dot(U2))  # 1*1

        # 오류역전파
        gradient = (y_val-out_node) * d_logistic_sig(hidden_node.dot(U2)) #1*1
        dU2 = -gradient*hidden_node.reshape(-1,1) #3*1
        gradient2 = gradient.dot(U2[1:].T) * d_logistic_sig(x_val.dot(U1)) #1*2
        dU1 = -gradient2*x_val.reshape(-1,1) #3*2

        # 가중치 갱신
        U2 += -learning_rate*dU2
        U1 += -learning_rate*dU1
    # 목적함수를 이용한 오차 확인
    if mse(y_val,out_node)<=1e-3:
        print(mse(y_val,out_node))
        break

# 테스트
for x_val, y_val, in zip(X,Y):
    hidden_node = np.insert(logistic_sig(x_val.dot(U1)),0,1) # 1*4
    out_node = logistic_sig(hidden_node.dot(U2))  # 1*2
    print(out_node)
