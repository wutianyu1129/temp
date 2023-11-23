from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective(x):
    return x[0]**2 + 9*x[1]**2

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: 2*x[0] + x[1] - 1},
    {'type': 'ineq', 'fun': lambda x: x[0] + 3*x[1] - 1},
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]}
]

# Initial guess
x0 = [0.5, 0.5]

# Solving the problem
solution = minimize(objective, x0, constraints=constraints)
x_opt = solution.x

# Generating data for contours and constraints
x = np.linspace(0, 1.5, 400)
y = np.linspace(0, 1.5, 400)
X, Y = np.meshgrid(x, y)
Z = X**2 + 9*Y**2

# Constraint lines
constraint1 = (1 - 2*x)/1
constraint2 = (1 - x)/3

# Plotting
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(x, constraint1, label=r'$2x_1 + x_2 \geq 1$')
plt.plot(x, constraint2, label=r'$x_1 + 3x_2 \geq 1$')
plt.fill_between(x, np.maximum(0, constraint1), constraint2, where=(constraint2>=constraint1) & (constraint2>=0), color='gray', alpha=0.3)
plt.scatter(x_opt[0], x_opt[1], color='red', label='Optimal Point')
plt.xlim([0, 1.5])
plt.ylim([0, 1.5])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Feasible Region with Objective Function Contours')
plt.grid(True)
plt.savefig('1.jpg')

x_opt
