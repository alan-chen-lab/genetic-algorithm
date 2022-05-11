import random
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
# SCH, d = 1

'''-------- param --------'''

population_size = 100
d = 1
max = 10**3
min = -10**3

'''-------- function --------'''


def function1(x):
    value = x**2
    return value


def function2(x):
    value = (x-2)**2
    return value


def evaluation(pop):
    # because of 2 objective functions
    fitness_values = np.zeros((len(pop), 2))
    pop = np.array(pop)
    # print('pop', pop)
    for i, chromosome in enumerate(pop):
        for j in range(2):
            if j == 0:      # objective 1
                fitness_values[i, j] = chromosome**2
            elif j == 1:    # objective 2
                fitness_values[i, j] = (chromosome - 2)**2

    return fitness_values


'''-------- fast_non_dominated_sort --------'''


def non_dominated_sorting(population_size, fitness_values):
    s, n = {}, {}  # 初始化 S set, len = pop_size # 初始化被支配的次數
    front, rank = {}, {}  # 初始化 to store rank, level set
    front[0] = []  # 一開始為空集合
    for p in range(population_size*2):  # for each p<P
        s[p] = []  # 第0次為空值(P支配次數)
        n[p] = 0  # 第0次為空值(被支配的次數)
        for q in range(population_size*2):
            # fitness_values[p][0] = value1, fitness_values[p][1] = value2
            if ((fitness_values[p][0] < fitness_values[q][0] and fitness_values[p][1] < fitness_values[q][1]) or (fitness_values[p][0] <= fitness_values[q][0] and fitness_values[p][1] < fitness_values[q][1])
                    or (fitness_values[p][0] < fitness_values[q][0] and fitness_values[p][1] <= fitness_values[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((fitness_values[p][0] > fitness_values[q][0] and fitness_values[p][1] > fitness_values[q][1]) or (fitness_values[p][0] >= fitness_values[q][0] and fitness_values[p][1] > fitness_values[q][1])
                  or (fitness_values[p][0] > fitness_values[q][0] and fitness_values[p][1] >= fitness_values[q][1])):
                n[p] = n[p]+1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in s[p]:
                n[q] = n[q]-1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front[i] = Q

    del front[len(front)-1]
    return front


'''--------calculate crowding distance function---------'''


def calculate_crowding_distance(front, fitness_values):

    distance = {m: 0 for m in front}
    for o in range(2):
        obj = {m: fitness_values[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)
        distance[sorted_keys[0]
                 ] = distance[sorted_keys[len(front)-1]] = 999999999999
        for i in range(1, len(front)-1):
            if len(set(obj.values())) == 1:
                distance[sorted_keys[i]] = distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]] = distance[sorted_keys[i]]+(
                    obj[sorted_keys[i+1]]-obj[sorted_keys[i-1]])/(obj[sorted_keys[len(front)-1]]-obj[sorted_keys[0]])

    return distance


'''----------selection----------'''


def selection(population_size, front, fitness_values, total_chromosome):
    # 族群數, level set, fitness values, parent+child_pop
    N = 0
    new_pop = []
    while N < population_size:
        for i in range(len(front)):
            N = N+len(front[i])
            if N > population_size:
                distance = calculate_crowding_distance(
                    front[i], fitness_values)
                sorted_cdf = sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop) == population_size:
                        break
                    new_pop.append(j)
                break
            else:
                new_pop.extend(front[i])

    population_list = []
    for n in new_pop:
        population_list.append(total_chromosome[n])

    return population_list, new_pop


'''==================== main code ===================='''
'''-------- initial population --------'''
solution = [[random.randint(min, max) for j in range(d)]  # solution = init_pop
            for i in range(population_size)]    # 20*1
function1_values = [[function1(solution[i][j])
                    for j in range(d)] for i in range(population_size)]  # 20*1
function2_values = [[function2(solution[i][j])
                    for j in range(d)] for i in range(population_size)]  # 20*1
print('solution\n = ', solution)
print('function1_values\n = ', function1_values)
print('function2_values\n = ', function2_values)

iteration = 200
for iter in range(iteration):  # loop
    '''-------- crossover --------'''
    # Whole arithmetic crossover
    parent_list = copy.deepcopy(solution)
    print('parent_list\n = ', parent_list)
    offspring_list = []
    # generate a random sequence to select the parent chromosome to crossover
    S = list(np.random.permutation(population_size))
    print('S\n = ', S)
    for m in range(int(population_size/2)):
        alpha = 0.5
        parent_1 = solution[S[2*m]][:]  # temp parent
        parent_2 = solution[S[2*m+1]][:]
        child_1 = alpha * parent_1[0] + (1 - alpha) * parent_2[0]
        child_2 = alpha * parent_2[0] + (1 - alpha) * parent_1[0]
        # append child chromosome to offspring list
        offspring_list.extend(([child_1], [child_2]))
    print('parent_1\n = ', parent_1)
    print('parent_2\n = ', parent_2)
    print('child_1\n = ', child_1)
    print('child_1\n = ', child_2)
    print('offspring_list\n = ', offspring_list)
    print('offspring_list_len\n = ', np.shape(offspring_list))

    '''--------mutatuon--------'''
    # Non-uniform mutation
    r = np.random.rand(1)
    b = 5
    T = iteration  # max_iteration
    number_of_time = []
    mutation_population = np.copy(offspring_list)
    gene = np.zeros((population_size, d))
    for i in range(population_size):
        gene[i, :] = np.random.randint(0, 2, d)
        for j in range(d):
            if gene[i, j] == 0:
                y0 = max - offspring_list[i][j]        # for y
                delta0 = y0*(1 - r**(1 - (iter+1)/float(T))**b)
                mutation_population[i][j] = offspring_list[i][j] + delta0
            elif gene[i, j] == 1:
                y1 = offspring_list[i][j] - (min)     # for y
                delta1 = y1*(1 - r**(1 - (iter+1)/float(T))**b)
                mutation_population[i][j] = offspring_list[i][j] - delta1
    mutation_population_list = mutation_population.tolist()
    print('mutation_population\n = ', mutation_population_list)

    '''--------fitness value-------------'''
    total_chromosome = copy.deepcopy(
        parent_list)+copy.deepcopy(mutation_population_list)  # combine parent and offspring chromosomes
    print('total_chromosome\n = ', total_chromosome)
    print('total_chromosome_len\n = ', np.shape(total_chromosome))  # 2N
    fitness_values = evaluation(total_chromosome)  # numpy
    print('fitness_values\n = ', fitness_values)
    print('fitness_values_len\n = ', np.shape(fitness_values))  # 2N * d

    '''-------non-dominated sorting-------'''
    # numpy to list
    fitness_values_list = fitness_values.tolist()
    print('fitness_values_list\n = ', fitness_values_list)
    front = non_dominated_sorting(population_size, fitness_values_list)
    print('front\n = ', front)

    '''----------selection----------'''
    population_list, new_pop = selection(
        population_size, front, fitness_values_list, total_chromosome)
    print('population_list\n = ', population_list)
    # print('new_pop\n = ', new_pop)

    solution = copy.deepcopy(population_list)

fitness_values = evaluation(population_list)  # numpy
print('Population\n = ',  np.array(population_list))
plt.scatter(fitness_values[:, 0], fitness_values[:, 1])
plt.xlabel('Objective function 1')
plt.ylabel('Objective function 2')
plt.legend(['pareto front'])
plt.title('MOGA of SCH')
plt.grid()
plt.show()
# sucess, all pop in accept error = [0, 2]
