import matplotlib.pyplot as plt
import math
import random
import numpy as np
import copy

"""
Target Function：
Length of the path that visits every city for exactly one time and goes back to the starting point
"""

    
def ga(chromosome_length, pop, iteration, cities):
    pc = 0.4 # probability of crossover
    pm = 0.4  # probability of mutation

    c_min = 20
    results = []
    # pop = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] for i in range(pop_size)]
    
    best_x = []
    best_y = -10000
    for i in range(iteration):
        obj_value = calc_obj_value(pop, chromosome_length, cities)  
        fit_value = calc_fit_value(obj_value, c_min)  
        best_individual, best_fit = find_best(pop, fit_value)  

        results.append([best_individual, best_fit])
        if (results[-1][1]>best_y):
            best_x = copy.deepcopy(results[-1][0][:])
            #print ("\nbest_x is changed to ", best_x)
            best_y = results[-1][1]
        #print("in iteration ", i, " best individual = ", best_individual, " and best_fit value = ", best_fit)
        #print("\nup to now best_x = ", best_x, " and best_y = ", best_y)
        #plt.plot(binary2decimal(best_individual, chromosome_length), fit_value)
        selection(pop, fit_value)  
        crossover(pop, pc)  
        mutation(pop, pm)  
    return results, (best_x, best_y)

def main():
    pop_size = 20
    chromosome_length = 6
    optimum_dict = {}
    iteration = 20
    pop = init_population(pop_size, chromosome_length)
    cities = init_city()
    n = 100
    optimum_list = []
    best_permutation_among_n = []
    shortest_length_among_n = 10000
    correct = 0
    for i in range(n):
        results, optimum = ga(chromosome_length, pop, iteration, cities)
        if (80-optimum[1] < 31.2):
            correct += 1
        if (80-optimum[1]<shortest_length_among_n):
            shortest_length_among_n = 80-optimum[1]
            best_permutation_among_n = copy.deepcopy(optimum[0])
        
        if (i==n-1):
            plot_iter_curve(iteration, [[results[i][0], 80-results[i][1]] for i in range(len(results))])
        
        optimum_list.append(80-optimum[1])
    for i in range(n):
        plt.scatter(i, optimum_list[i], c='r', s=10)
    plt.show()
    print("shortest path length: ", 80-calc_target_function(cities, best_permutation_among_n, chromosome_length), shortest_length_among_n)
    print("best permutation: ", best_permutation_among_n)
    print("accuracy: ", correct, " / ", n)




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


def binary2decimal(binary, chromosome_length):
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 2 ** j
    t -= 63
    return t


def init_population(pop_size, chromosome_length):
    pop = [np.random.permutation(chromosome_length) for i in range(pop_size)]
    return pop

def init_city():
    cities = []
    cities.append((3,2))
    cities.append((5,8))
    cities.append((4,12))
    cities.append((6,7))
    cities.append((11,13))
    cities.append((9,10))
    return cities

# decode a binary string to its corresponding decimal value
def decode_chromosome(pop, chromosome_length, upper_limit):
    X = []
    for ele in pop:
        temp = 0
        for i, coff in enumerate(ele):

            temp += coff * (2 ** i)

        temp -= 63
        X.append(temp)
    
    return X

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

def calc_obj_value(pop, chromosome_length, cities):
    obj_value = []
    for permutation in pop:
        obj_value.append(calc_target_function(cities, permutation, chromosome_length))
    return obj_value


def calc_fit_value(obj_value, c_min):
    fit_value = []

    c_min
    for value in obj_value:
        if value > c_min:
            temp = value
        else:
            temp = 0.
        fit_value.append(temp)
    
    return fit_value

#find the best fitness value within a population as well as its corresponding gene encoding
def find_best(pop, fit_value):

    best_individual = pop[0]

    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = copy.deepcopy(pop[i])
    return best_individual, best_fit


def cum_sum(p_fit_value):

    temp = p_fit_value[:]
    for i in range(len(temp)):
        p_fit_value[i] = (sum(temp[:i + 1]))

def selection(pop, fit_value):
    p_fit_value = []
    # total fitness value
    total_fit = sum(fit_value)
    # normalization
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit)

    cum_sum(p_fit_value)
    pop_len = len(pop)
 
    newpop = []

    while (len(newpop) < pop_len):
        r = random.random()
        for i in range(pop_len):
            if (r<p_fit_value[i]):
                newpop.append(pop[i])
                break
    pop = newpop[:]


def crossover(pop, pc):
 
    pop_len = len(pop)
    for i in range(pop_len - 1):
        # crossover between two consecutive individuals in the population list
        if (i%2 == 1):
            continue
        if (random.random() < pc):

            c = random.randint(0, len(pop[0])-1)
            pair = (pop[i][c], pop[i+1][c])

            for j in range(len(pop[i])):
                if (pop[i][j] == pair[0]):
                    pop[i][j] = pair[1]
                    continue
                if (pop[i][j] == pair[1]):
                    pop[i][j] = pair[0]
            for j in range(len(pop[i+1])):
                if (pop[i+1][j] == pair[0]):
                    pop[i+1][j] = pair[1]
                    continue
                if (pop[i+1][j] == pair[1]):
                    pop[i+1][j] = pair[0]



def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    # randomly choose two cities and change their order
    for i in range(px):
        if (random.random() < pm):
            m = random.randint(0, py - 1)
            n = random.randint(0, py - 1)
            while (m==n):
                m = random.randint(0, py - 1)
                n = random.randint(0, py - 1)
            temp = pop[i][m]
            pop[i][m] = pop[i][n]
            pop[i][n] = temp

if __name__ == '__main__':
    main()
