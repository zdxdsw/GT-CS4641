import matplotlib.pyplot as plt
import math
import random
import numpy as np
import copy
"""
Target Function：
Length of the path that visits every city for exactly one time and goes back to the starting point
"""

def sa(chromosome_length, cities):
    T_init = 300  # initial temperature
    alpha = 0.985  # controlling factor for temperature reduction
    T_min = 0.01  # final temperature (defines stopping criteria)
    T = T_init
    x = np.random.permutation(chromosome_length)  # initialize x to be a random permutation in range(num_cities)
    y = calc_target_function(cities, x, chromosome_length) # compute function value for initial x
    results = []  # a list to store (x，y)
    x_best = copy.deepcopy(x)
    y_best = y

    while T > T_min:
        c = random.randint(1, chromosome_length-2)
        x_new = []
        i=c+1
        while (i<chromosome_length):
            x_new.append(x[i])
            i += 1
        j=0
        while (j<=c):
            x_new.append(x[c-j])
            j += 1

        y_new = calc_target_function(cities, x_new, chromosome_length)
        de = y - y_new 
        if (de<0): # y_old < y_new: accept new value
            #flag = 1
            x = copy.deepcopy(x_new)
            y = y_new
            if y > y_best:
                x_best = copy.deepcopy(x)
                y_best = y
        elif (math.exp(-de/T) > random.random()):
            # accept new value with probability = exp(-de/T)
            #flag = 1  
            x = copy.deepcopy(x_new)
            y = y_new
            if y > y_best:
                x_best = copy.deepcopy(x)
                y_best = y
        #print("\nappend x = ", x, " y = ", y)
        results.append((x, y))
        T *= alpha
    optimum = (copy.deepcopy(x_best),y_best)
    return results, optimum

def main():
    chromosome_length = 6
    cities = init_city()
    n = 100
    optimum_list = []
    best_permutation_among_n = []
    shortest_length_among_n = 10000
    correct = 0
    for i in range(n):
        results, optimum = sa(chromosome_length, cities)
        if (80-optimum[1] < 31.2):
            correct += 1
        if (80-optimum[1]<shortest_length_among_n):
            shortest_length_among_n = 80-optimum[1]
            best_permutation_among_n = copy.deepcopy(optimum[0])
        
        if (i==n-1):
            #plot_path_map(chromosome_length, cities, optimum[0])
           # plot_iter_curve(len(results), results)
            plot_iter_curve(len(results), [[results[i][0], 80-results[i][1]] for i in range(len(results))])
        
        optimum_list.append(80-optimum[1])
    
    for i in range(n):
        plt.scatter(i, optimum_list[i], c='r', s=10)
    plt.show()
    print("shortest path length: ", 80-calc_target_function(cities, best_permutation_among_n, chromosome_length), shortest_length_among_n)
    print("best permutation: ", best_permutation_among_n)
    print("accuracy: ", correct, " / ", n)
    #plot_path_map(chromosome_length, cities, best_permutation_among_n)

def init_city():
    cities = []
    cities.append((3,2))
    cities.append((5,8))
    cities.append((4,12))
    cities.append((6,7))
    cities.append((11,13))
    cities.append((9,10))
    return cities

def calculate_distance(point1, point2):
    dx = math.fabs(point1[0] - point2[0])
    dy = math.fabs(point1[1] - point2[1])
    return math.sqrt(dx*dx + dy*dy)

def calc_target_function(cities, permutation, chromosome_length):
    path_len = 0
    for i in range(chromosome_length-1):
        path_len += calculate_distance(cities[permutation[i]], cities[permutation[i+1]])
    path_len += calculate_distance(cities[permutation[chromosome_length-1]], cities[permutation[0]])
    return (80-path_len)

def plot_path_map(chromosome_length, cities, permutation):
    X = []
    Y = []
    #print("in plot_path_map function: ", permutation)
    for i in range(chromosome_length):
        X.append(cities[permutation[i]][0])
        Y.append(cities[permutation[i]][1])
    X.append(cities[permutation[0]][0])
    Y.append(cities[permutation[0]][1])
    for i in range(chromosome_length):
        plt.scatter(cities[i][0], cities[i][1], c='r', s=10*i)
    plt.plot(X, Y)
    plt.show()

def plot_iter_curve(iter, results):
    X = [i for i in range(iter)]
    Y = [results[i][1] for i in range(iter)]
    plt.plot(X, Y)
    plt.show()



if __name__ == '__main__':
    # for i in range(100):
    main()

