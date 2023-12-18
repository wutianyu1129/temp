# 导入所需的库
from scipy.optimize import minimize
from scipy.signal import convolve
import numpy as np
import scipy.linalg as la
from numpy.random import RandomState
from scipy import signal
import numpy as np
from scipy import optimize, fft
import matplotlib.pyplot as plt

# Parameters
T = 400
k = 20
N = T+k
p=0.1
sigma = 0.0001
#  Random Model with fixed seed
rn = RandomState(364)

w_true = rn.rand(k)
index =np.argmax(np.abs(w_true))
w_true = np.roll(w_true,-index)
w_true = w_true/w_true[0]


x_true = rn.randn(T)
x_true = rn.binomial(1,p,np.shape(x_true))*x_true
y_true = np.real(np.fft.ifft( np.fft.fft(x_true,N)/np.fft.fft(w_true,N),N))
y= y_true[k:T+k]+ sigma* rn.randn(T)

def inverse_ker(w, len=N):
	w_inv = np.real(np.fft.ifft(1/np.fft.fft(w,len),len))
	return w_inv

# 定义 l1 范数计算函数
def l1_norm(w, y, T, k):
    # 计算 w 和 y 的卷积
    x = convolve(w, y, mode='full')
    # 只考虑 x[k:T]
    x_truncated = x[k-1:T]
    # 计算 l1 范数
    return np.sum(np.abs(x_truncated))

# 定义优化问题的约束条件：w 的和为 1
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# 初始化 w
w_initial = np.random.rand(k)

# 解决优化问题
result = minimize(l1_norm, w_initial, args=(y, T, k), constraints=constraint, method='SLSQP')

# 检查优化是否成功
if result.success:
    w_optimal = result.x
    # 通过最优 w 和 y 的卷积得到稀疏信号 x
    x_optimal = convolve(w_optimal, y, mode='full')[k-1:T]
else:
    w_optimal = x_optimal = None
    print("优化失败:", result.message)

# 计算逆核函数
def compute_inverse_kernel(w, N):
    Fw = fft.fft(w, N)
    inverse_Fw = 1 / np.where(Fw != 0, Fw, 1)
    return np.real(fft.ifft(inverse_Fw))

# 计算逆核 w^-1
w_inverse = compute_inverse_kernel(w_optimal, N)

# 绘制结果图
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.stem(w_optimal, use_line_collection=True)
plt.title('最优滤波器 w')
plt.xlabel('索引')
plt.ylabel('幅度')

plt.subplot(4, 1, 2)
plt.plot(x_optimal)
plt.title('稀疏信号 x')
plt.xlabel('时间')
plt.ylabel('幅度')

plt.subplot(4, 1, 3)
plt.plot(y)
plt.title('观测 y')
plt.xlabel('时间')
plt.ylabel('幅度')

plt.subplot(4, 1, 4)
plt.stem(w_inverse, use_line_collection=True)
plt.title('逆核 w^-1')
plt.xlabel('索引')
plt.ylabel('幅度')

plt.tight_layout()
plt.show()
plt.savefig('1.jpg')

