import mlrose
import numpy as np
import matplotlib.pyplot as plt

"""
Target Function：
Length of the path that visits every city for exactly one time and goes back to the starting point
"""

# Create list of city coordinates
coords_list = [(3, 2), (5, 8), (4, 12), (6, 7), (11, 13), (9, 10)]

# Define optimization problem object
problem = mlrose.TSPOpt(length = 6, coords = coords_list, maximize=False)

# Set random seed
#np.random.seed(2)
n=100
optimum_list = []
correct = 0
for i in range(n):
    best_x, best_y, next_fitness_list= mlrose.mimic(problem, pop_size=50, keep_pct=0.8, max_attempts=30, max_iters=100)
    optimum_list.append(best_y)
    if (best_y < 31.2):
        correct += 1
print("accuracy: ", correct, " / ", n)
for i in range(n):
    plt.scatter(i, optimum_list[i], c='r', s=10)
plt.show()
X = [i for i in range(len(next_fitness_list))]
Y = [next_fitness_list[i] for i in range(len(next_fitness_list))]
plt.plot(X, Y)
plt.show()

print(best_x)
print(best_y)
