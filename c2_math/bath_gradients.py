import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = 2*np.random.rand(100,1)
y = 5 + 9*X + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X]

plt.figure(figsize=(10,4))
plt.plot(X,y,"b.")

learning_rate = 0.2   # 학습률 설정
n_iter = 1000           # 반복 횟수 설정
m = 100                 # 샘플 수

theta = np.random.randn(2,1)   # 정규분포를 따르는 무작위 초기값 설정


for iter in range(n_iter):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)  # 배치 경사하강법
    theta = theta - learning_rate*gradients          # next step


X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
print(X_new_b.dot(theta))

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 20])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42)
theta = np.random.randn(2,1)  # 무작위 초기값 설정

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.4)

plt.show()
