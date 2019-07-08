import matplotlib.pyplot as plt
import math
import random
import copy


"""
Target Function：
Sum of the absolute difference for each array element between
the current state vector and the “target vector” [3,3,3,3,3,3]
"""


def sa(target_string, string_length):
    T_init = 50  # initial temperature
    alpha = 0.998  # controlling factor for temperature reduction
    T_min = 0.01  # final temperature (defines stopping criteria)
    T = T_init
    x = [random.randint(1, string_length) for i in range(string_length)]
    y = calc_target_function(target_string, x, string_length) # compute function value for initial x
    results = []  # a list to store (x，y)
    x_best = x
    y_best = y

    while T > T_min:
        i = random.randint(0,string_length-1)
        if (x[i] == 1):
            x_new = copy.deepcopy(x)
            x_new[i] = 2
        elif (x[i] == string_length):
            x_new = copy.deepcopy(x)
            x_new[i] = string_length-1
        else:
            x_new = copy.deepcopy(x)
            if (random.random() < 0.5):
                x_new[i] -= 1
            else: x_new[i] += 1
        
        y_new = y = calc_target_function(target_string, x_new, string_length)
        de = y - y_new 
        if (de<0): # y_old < y_new: accept new value

            x = copy.deepcopy(x_new)
            y = y_new
            if y > y_best:
                x_best = copy.deepcopy(x)
                y_best = y
        elif (math.exp(-de/T) > random.random()):
 
            x = copy.deepcopy(x_new)
            y = y_new
            if y > y_best:
                x_best = copy.deepcopy(x)
                y_best = y
                   
        results.append((copy.deepcopy(x), y))
        T *= alpha

    optimum = (copy.deepcopy(x_best),y_best)
    return results, optimum
def main():
    string_length = 5
    target_string = init_target(string_length)
    optimum_list = []
    #optimum_dict = {}
    n = 100
    correct_times = 0
    for i in range(n):
        results, optimum = sa(target_string, string_length)
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
    return [3,3,3,3,3]

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

