import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def objective(x, P, q, r):
    return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x) + r

# 目标函数的参数
P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
q = np.array([-22, -14.5, 13])
r = 1

# 初始猜测
x0 = np.array([0, 0, 0])

# 变量的界限
bounds = [(-1, 1), (-1, 1), (-1, 1)]

# 使用 'SLSQP' 算法进行优化
result = minimize(objective, x0, args=(P, q, r), method='SLSQP', bounds=bounds)

# 输出结果
print("Optimal solution:", result.x)
print("Objective function value at optimal solution:", result.fun)

import time
import numpy as np

def benchmark():
    N = 10000000
    A = np.random.rand(N)
    B = np.random.rand(N)

    start_time = time.time()
    C = A * B  # 一个简单的向量乘法操作
    end_time = time.time()

    return end_time - start_time

# 运行跑分
time_taken = benchmark()
print(f"Time taken for benchmark: {time_taken:.4f} seconds")
