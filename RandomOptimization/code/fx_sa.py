import matplotlib.pyplot as plt
import math
import random

"""
Target Function：
f(x) = 8[sin(0.6x)+cos(1.4x)+e^(cos(0.2x))]
Domain of x: integer in the interval [-63, 64]
"""


def sa():
    T_init = 100  # initial temperature
    alpha = 0.99  # controlling factor for temperature reduction
    T_min = 0.01  # final temperature (defines stopping criteria)
    T = T_init
    x = random.randrange(-63, 65, 1)  # initialize x to be a random integer in within the interval [-63, 64]
    y = 8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) # compute function value for initial x
    results = []  # a list to store (x，y)
    x_best = x
    y_best = y
    explore_range_float = 10.0
    explore_range = math.ceil(explore_range_float)
    while T > T_min:
        i = random.randrange(0,2*explore_range +1,1)
        x_new = x+i-explore_range
        if (x_new<-63 or x_new>64):
            continue
        y_new = 8 * math.sin(0.06 * x_new) + 8 * math.cos(0.14 * x_new) + 8 * math.exp(math.cos(0.2*x_new))
        de = y - y_new 
        if (de<0): # y_old < y_new: accept new value
            #flag = 1
            x = x_new
            y = y_new
            if y > y_best:
                x_best = x
                y_best = y
        elif (math.exp(-de/T) > random.random()):
            # accept new value with probability = exp(-de/T)
            #flag = 1  
            x = x_new
            y = y_new
            if y > y_best:
                x_best = x
                y_best = y
                   
        results.append((x, y))
        T *= alpha
        explore_range_float *= alpha
        explore_range = math.ceil(explore_range_float)
        #explore_range_integer = math.ceil(explore_range)
    optimum = (x_best,y_best)
    return results, optimum
def main():
    optimum_dict = {}
    optimum_list = []
    n=100
    for i in range(n):
        results, optimum = sa()
        optimum_list.append(optimum[1])
        if (optimum[0] in optimum_dict):
            optimum_dict[optimum[0]] += 1
        else:
            optimum_dict[optimum[0]] = 1
        if (i==99):
            plot_final_result(results)
            plot_iter_curve(results)
    correct_times = 0
    if 0 in optimum_dict:
        correct_times = optimum_dict[0]
    print("accuracy: ", correct_times, "/100")
    print(optimum_dict)
    for i in range(n):
        plt.scatter(i, optimum_list[i], c='r', s=10)
    plt.show()

def plot_iter_curve(results):
    X = [i for i in range(len(results))]
    Y = [results[i][1] for i in range(len(results))]
    plt.plot(X, Y)
    plt.show()

def plot_final_result(results):
    X1 = [i for i in range(-63, 65, 1)]
    Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
    plt.plot(X1, Y1)
    plt.scatter(results[-1][0], results[-1][1], c='r', s=10)
    plt.show()

if __name__ == '__main__':
    # for i in range(100):
    main()

