import matplotlib.pyplot as plt
import math
import random
import copy

"""
Target Function：
f(x) = 8[sin(0.6x)+cos(1.4x)+e^(cos(0.2x))]
Domain of x: integer in the interval [-63, 64]
"""


    
def ga(iteration):
    pop_size = 20  
    upper_limit = 10  
    chromosome_length = 7 
    pc = 0.6 
    pm = 0.6  
    c_min = 10
    results = [] 
    pop = init_population(pop_size, chromosome_length)
    results_dict = {}
    best_x = -100
    best_y = -100
    for i in range(iteration):
        obj_value = calc_obj_value(pop, chromosome_length, upper_limit) 
        fit_value = calc_fit_value(obj_value, c_min)

        best_individual, best_fit = find_best(pop, fit_value)  
        results.append([binary2decimal(best_individual, chromosome_length), best_fit])
        if (results[-1][1]>best_y):
            best_x = results[-1][0]
            best_y = results[-1][1]
       
        selection(pop, fit_value) 
        crossover(pop, pc) 
        mutation(pop, pm) 

    return results, (results[-1][0], results[-1][1])
def main():
    optimum_dict = {}
    iteration = 500
    optimum_list = []
    n=100
    for i in range(n):
        print(i)
        results, optimum = ga(iteration)
        optimum_list.append(copy.deepcopy(optimum))
        if (optimum[0] in optimum_dict):
            optimum_dict[optimum[0]] += 1
        else:
            optimum_dict[optimum[0]] = 1
        if (i==99):
            plt.scatter(results[-1][0], results[-1][1], s=5, c='r')
            X1 = [i for i in range(-63, 65, 1)]
            Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
            plt.plot(X1, Y1)
            plt.show()
            
            plot_iter_curve(iteration, results)
    correct_times = 0
    if 0 in optimum_dict:
        correct_times = optimum_dict[0]
    print("accuracy: ", correct_times, "/100")
    import operator
    optimum_dict = sorted(optimum_dict.items(), key=lambda kv: -kv[1])
    for i in range(n):
        plt.scatter(optimum_list[i][0], optimum_list[i][1], c='r', s=10)
    X1 = [i for i in range(-63, 65, 1)]
    Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
    plt.plot(X1, Y1)
    plt.show()
    print(optimum_dict)



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
    # 形如[[0,1,..0,1],[0,1,..0,1]...]
    pop = [[random.randint(0, 1) for i in range(chromosome_length)] for j in range(pop_size)]
    return pop



def decode_chromosome(pop, chromosome_length, upper_limit):
    X = []
    for ele in pop:
        temp = 0
        for i, coff in enumerate(ele):
            temp += coff * (2 ** i)

        temp -= 63
        X.append(temp)
    
    return X


def calc_obj_value(pop, chromosome_length, upper_limit):
    obj_value = []
    X = decode_chromosome(pop, chromosome_length, upper_limit)
    for x in X:

        obj_value.append(8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)))
    X1 = [i for i in range(-63, 65, 1)]
    Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
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
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1

if __name__ == '__main__':
    main()
