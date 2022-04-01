import numpy as np
import random
from random import choice
import math
import copy
import timeit
import matplotlib.pyplot as plt

def initial_array(populationSize, d):
    initial_array = []
    for i in range(populationSize):
        for j in range(d):
            initial_value = random.randint(-100, 100)
            initial_array.append(initial_value)
    initial_pop = np.reshape(initial_array, (populationSize, d)).astype('f')  # 初始化族群
    return initial_pop

def unimodel_function(initial_pop):
    pop, d = initial_pop.shape
    total = np.zeros((pop, 1), dtype=np.float32)
    for a in range(pop):
        total[a, :] = sum(np.cumsum(initial_pop[a, :]) ** 2)    # 題目(unimodel)
    # print("fitness value:\n", total)
    return total

def selection(initial_pop, total):
    pop, d = initial_pop.shape
    temp_child_fitness = []
    k_number = []
    after_select_pop = np.zeros((pop, d), dtype=np.float32)
    while (len(temp_child_fitness) < pop):      # 使用Tournament Selection(k = 2)
        array = np.arange(pop)                  # without replacement
        k1 = choice(array)
        k2 = choice(array)
        if (total[k1, :] >= total[k2, :]):
            temp_child_fitness.append(total[k2, :])
            k_number.append(k2)
        elif (total[k1, :] < total[k2, :]):
            temp_child_fitness.append(total[k1, :])
            k_number.append(k1)
        for i in range(len(k_number)):
            after_select_pop[i, :] = initial_pop[k_number[i], :]
    return k_number, after_select_pop

def crossover(alpha, initial_pop, after_select_pop):    # Whole arithmetic crossover
    pop, d = initial_pop.shape
    array = np.arange(pop)
    a = choice(array)
    b = choice(array)
    select_parent_1 = after_select_pop[a, :]
    select_parent_2 = after_select_pop[b, :]
    select_parent_stack = np.vstack((select_parent_1, select_parent_2))  # 2*1
    after_crossover_pop = np.zeros((2, d), dtype=np.float32)
    # print('select_parent_stack:\n', select_parent_stack)
    for i in range(d):
        child_1 = alpha * select_parent_stack[:, i][0] + (1 - alpha) * select_parent_stack[:, i][1]
        after_crossover_pop[:, i][0] = child_1
        child_2 = alpha * select_parent_stack[:, i][1] + (1 - alpha) * select_parent_stack[:, i][0]
        after_crossover_pop[:, i][1] = child_2
    # print('after_crossover_pop:\n', after_crossover_pop)
    return after_crossover_pop

def whole_crossover(alpha, initial_pop, after_select_pop):  # Whole arithmetic crossover
    pop, d = initial_pop.shape
    a = []
    c = 10
    for i in range(c):
        after_crossover_pop = crossover(alpha, initial_pop, after_select_pop)
        a.append(after_crossover_pop)
    crossover_all = np.reshape(a, (c*2, d)).astype('f')
    # print('crossover_all:\n', crossover_all)
    return crossover_all

def mutation(initial_pop, after_crossover_pop, iteration):  # Non-uniform mutation
    pop, d = initial_pop.shape
    b = 5
    T = 10000
    r = np.random.rand(1)                            # for random r
    mutation_pop = np.zeros((pop, d), dtype=np.float32)
    gene = np.zeros((pop, d))
    for i in range(pop):
        gene[i, :] = np.random.randint(0, 2, d)      
        for j in range(d):
            if(gene[i, j] == 0):
                y0 = 100 - after_crossover_pop[i, j]        # for y
                delta0 = y0*(1 - r**(1 - (iteration+1)/float(T))**b)
                mutation_pop[i, j] = after_crossover_pop[i, j] + delta0
            elif(gene[i, j] == 1):
                y1 = after_crossover_pop[i, j] - (-100)     # for y
                delta1 = y1*(1 - r**(1 - (iteration+1)/float(T))**b)
                mutation_pop[i, j] = after_crossover_pop[i, j] - delta1
    return mutation_pop

population_size = 20
d = 30
iteration = 10000
optimalValues = []
generation_time = []
iteration_record = []
optimal_v = []
min_generation_time = []
init_population = initial_array(population_size, d)
# print('initial_population:\n', init_population)
for iter in range(iteration):
    fitness_value = unimodel_function(init_population)
    # print('fitness_value:\n', fitness_value)
    k_number, after_selection_population = selection(init_population, fitness_value)
    # print('k_number:\n', k_number)
    # print('after_selection_population:\n', after_selection_population)
    # alpha = np.random.rand(1)
    alpha = 0.5
    cross = crossover(alpha, init_population, after_selection_population)
    c_all = whole_crossover(alpha, init_population, after_selection_population)
    # print('crossover:\n', crossover)
    mutation_ = mutation(init_population, c_all, iter)
    # print('mutation_:\n', mutation_)
    init_population = mutation_
    final_fitness_value = unimodel_function(mutation_)
    # print('final_fitness_value:\n', final_fitness_value)
    optimalValues.append(np.min(list(final_fitness_value)))
    generation_time.append(iter)
    if (iter == 999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 1999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 2999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 3999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 4999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 5999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 6999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 7999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 8999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))
    elif (iter == 9999):
        iteration_record.append(iter)
        optimal_v.append(np.min(optimalValues))

optimalValue = np.min(optimalValues)
for i in range(iteration):
    a = optimalValues[i]
    if (optimalValue == a):
        min_generation_time.append(i)
print('迭代第幾代有最佳fitness value:', min_generation_time[0])
print('最佳fitness value(min), Acceptance: 0~100 :', optimalValue)
# 2d plot
plt.subplot(2, 1, 1)
plt.plot(generation_time, optimalValues)
plt.ylabel("fitness values")
plt.xlabel("generation")
plt.subplot(2, 1, 2)
plt.plot(iteration_record, optimal_v)
plt.ylabel("fitness values")
plt.xlabel("generation")
plt.show()

