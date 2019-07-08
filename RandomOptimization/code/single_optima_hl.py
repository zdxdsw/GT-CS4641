import matplotlib.pyplot as plt
import math
import random
import copy

"""
Target Function：
Sum of the absolute difference for each array element between
the current state vector and the “target vector” [3,3,3,3,3,3]
"""


def hl(target_string, string_length):
    iteration = 10
    x = [random.randint(1, string_length) for i in range(string_length)]
    y = calc_target_function(target_string, x, string_length) # compute function value for initial x
    results = []  # a list to store (x，y)
    results.append((copy.deepcopy(x), y))

    # try to climb at most <iteration> times
    for t in range(iteration):
        climb = 0
        x_best = copy.deepcopy(x)
        y_best = y
        for i in range(string_length):
            if (x[i] == 1):
                x_new = copy.deepcopy(x)
                x_new[i] = 2
                y_new = calc_target_function(target_string, x_new, string_length) # compute function value for initial x
                if (y_new > y_best):
                    climb = 1
                    x_best = copy.deepcopy(x_new)
                    y_best = y_new
            elif (x[i] == string_length):
                x_new = copy.deepcopy(x)
                x_new[i] = string_length-1
                y_new = calc_target_function(target_string, x_new, string_length) # compute function value for initial x
                if (y_new > y_best):
                    climb = 1
                    x_best = copy.deepcopy(x_new)
                    y_best = y_new
            else:
                x_new = copy.deepcopy(x)
                x_new[i] += 1
                y_new = calc_target_function(target_string, x_new, string_length) # compute function value for initial x
                if (y_new > y_best):
                    climb = 1
                    x_best = copy.deepcopy(x_new)
                    y_best = y_new
                
                x_new = copy.deepcopy(x)
                x_new[i] -= 1
                y_new = calc_target_function(target_string, x_new, string_length) # compute function value for initial x
                if (y_new > y_best):
                    climb = 1
                    x_best = copy.deepcopy(x_new)
                    y_best = y_new
                
        if (climb == 0): break
        else:
            x = copy.deepcopy(x_best)
            y = y_best
            results.append((copy.deepcopy(x), y))

    return results, (copy.deepcopy(x), y)
def main():
    string_length = 5
    target_string = init_target(string_length)
    optimum_list = []
    #optimum_dict = {}
    n = 100
    correct_times = 0
    for i in range(n):
        results, optimum = hl(target_string, string_length)
        optimum_list.append(optimum[1])
        if (optimum[1] == 20):
            correct_times += 1
        if (i==n-1):
            plot_iter_curve(results, target_string, string_length)
            
    print("accuracy: ", correct_times, "/100")
    for i in range(n):
        plt.scatter(i, 20-optimum_list[i], c='r', s=10)
    plt.show()


def init_target(string_length):
    #return [random.randint(1, string_length) for i in range(string_length)]
    return [3,3,3,3,3,3]

def calc_target_function(target_string, x, string_length):
    y = 20
    for i in range(string_length):
        y -= math.fabs(x[i] - target_string[i])
    return y

def plot_iter_curve(results, target_string, string_length):
    X = [i for i in range(len(results))]
    Y = [results[i][1] for i in range(len(results))]
    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    # for i in range(100):
    main()

