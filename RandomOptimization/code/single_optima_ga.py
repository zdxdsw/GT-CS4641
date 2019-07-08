import matplotlib.pyplot as plt
import math
import random
import copy
import numpy as np

"""
Target Function：
Sum of the absolute difference for each array element between
the current state vector and the “target vector” [3,3,3,3,3,3]
"""

    
def ga(iteration, target_string, string_length):
    pop_size = 400 
    chromosome_length = string_length 
    pc = 0.4 # probability of crossover
    pm = 0.6  # probability of mutation
    c_min = 0
    results = []  
    pop = init_population(pop_size, chromosome_length)
    results_dict = {}
    best_x = -100
    best_y = -100
    for i in range(iteration):
        obj_value = calc_obj_value(pop, target_string, chromosome_length) 
        fit_value = calc_fit_value(obj_value, c_min)  
        
        best_individual, best_fit = find_best(pop, fit_value) 
        results.append([best_individual, best_fit])
        if (results[-1][1]>best_y):
            best_x = copy.deepcopy(results[-1][0])
            best_y = results[-1][1]
        
        selection(pop, fit_value)  
        crossover(pop, pc)  
        mutation(pop, pm)  
    return results, (copy.deepcopy(results[-1][0]), results[-1][1])
def main():
    string_length = 5
    target_string = init_target(string_length)
    optimum_list = []
    iteration = 200
    #optimum_dict = {}
    n = 50
    correct_times = 0
    for i in range(n):
        results, optimum = ga(iteration, target_string, string_length)
        optimum_list.append(optimum[1])

        if (optimum[1] == 20):
            correct_times += 1
        if (i==n-1):
            plot_iter_curve(results, target_string, string_length)
            
    print("accuracy: ", correct_times, "/100")
    for i in range(n):
        plt.scatter(i, 20-optimum_list[i], c='r', s=10)
    plt.show()


def plot_obj_func():
    """y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)"""
    X1 = [i for i in range(-63, 65, 1)]
    Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
    plt.plot(X1, Y1)
    plt.show()


def plot_currnt_individual(X, Y):
    X1 = [i for i in range(-63, 65, 1)]
    Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
    plt.plot(X1, Y1)
    plt.scatter(X, Y, c='r', s=5)
    plt.show()


def plot_iter_curve(results, target_string, string_length):
    X = [i for i in range(len(results))]
    Y = [results[i][1] for i in range(len(results))]
    plt.plot(X, Y)
    plt.show()


def init_population(pop_size, chromosome_length):
    # 形如[[0,1,..0,1],[0,1,..0,1]...]
    pop = [[random.randint(1, chromosome_length) for i in range(chromosome_length)] for j in range(pop_size)]
    return pop


def init_target(string_length):
    #return [random.randint(1, string_length) for i in range(string_length)]
    return [3,3,3,3,3]

def calc_target_function(target_string, x, string_length):
    y = 20
    for i in range(string_length):
        y -= math.fabs(x[i] - target_string[i])
    return y

def calc_obj_value(pop, target_string, string_length):
    obj_value = []
    for p in pop:
        obj_value.append(
            calc_target_function(target_string, p, string_length))
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


def find_best(pop, fit_value):
    best_individual = []
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return best_individual, best_fit


def cum_sum(p_fit_value):

    temp = p_fit_value[:]
    for i in range(len(temp)):
        p_fit_value[i] = (sum(temp[:i + 1]))


def selection(pop, fit_value):
    p_fit_value = []
    total_fit = sum(fit_value) 
    for i in range(len(fit_value)):
        p_fit_value.append(fit_value[i] / total_fit) # normalization

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

        if (i%2 == 1):
            continue
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i + 1][cpoint:len(pop[i])])
            temp2.extend(pop[i + 1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1[:]
            pop[i + 1] = temp2[:]


def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])

    
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] += 1
            elif (pop[i][mpoint] == py):
                pop[i][mpoint] -= 1
            else:
                if (random.random()<0.5):
                    pop[i][mpoint] += 1
                else: pop[i][mpoint] -= 1

if __name__ == '__main__':
    main()
