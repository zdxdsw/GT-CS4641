import mlrose
import math
import matplotlib.pyplot as plt


"""
Target Function：
Sum of the absolute difference for each array element between
the current state vector and the “target vector” [3,3,3,3,3,3]
"""

def init_target(string_length):
    #return [random.randint(1, string_length) for i in range(string_length)]
    return [3,3,3,3,3]

def calc_target_function(x):
    string_length = 5
    target_string = [3,3,3,3,3]
    y = 0
    for i in range(string_length):
        y += math.fabs(x[i] - target_string[i])
    return y

# Initialize custom fitness function object
f = mlrose.CustomFitness(calc_target_function, "discrete")
problem = mlrose.DiscreteOpt(length = 5, fitness_fn = f, maximize = False, max_val = 5)
optimum_list = []
total = 0
n=50
for i in range(n):
    best_x, best_y, next_fitness_list= mlrose.mimic(problem, pop_size=300, keep_pct=0.8, max_attempts=50, max_iters=20)
    if (best_y==0):
        total += 1
    optimum_list.append(best_y)
        
X = [i for i in range(len(next_fitness_list))]
Y = [next_fitness_list[i] for i in range(len(next_fitness_list))]
plt.plot(X, Y)
plt.show()
for i in range(n):
    plt.scatter(i, optimum_list[i], c='r', s=10)
plt.show()
print(best_x, best_y)
print("accuracy: ", total/100.0)
