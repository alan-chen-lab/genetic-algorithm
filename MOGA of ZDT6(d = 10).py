import random
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
# ZDT6, d = 10

'''-------- param --------'''

population_size = 100
d = 10
max = 1
min = 0

'''-------- function --------'''


def f1(x):  # if x1
    ans_f1_ = []
    for i in range(population_size):
        for j in range(d):
            if j == 0:
                ans_f1 = [1 - math.exp(-4 * x[i][j]) *
                          (math.sin(6 * math.pi * x[i][j]))**6]
                ans_f1_.append(ans_f1)
    return ans_f1_


def g_x(x):  # if x2...
    ans_g_x = []
    for i in range(population_size):
        temp = []
        for j in range(d):
            if j != 0:
                temp.append(x[i][j])
        # print(temp)
        ans = 1 + 9 * (sum(temp) / (d - 1)) ** 0.25
        ans_g_x.append([ans])
    return ans_g_x


def f2(function1_values, g_x_values):
    ans_f2_ = []
    for i in range(population_size):
        for j in range(1):
            ans_f2 = g_x_values[i][j] * \
                (1 - (function1_values[i][j] / g_x_values[i][j]) ** 2)
        ans_f2_.append([ans_f2])
    return ans_f2_

# ------------------------------------------for fitness value 2N * d


def f1_2(x):  # if x1
    ans_f1_2 = []
    for i in range(2*population_size):
        for j in range(d):
            if j == 0:
                ans_f1_temp = [1 - math.exp(-4 * x[i][j]) *
                               (math.sin(6 * math.pi * x[i][j])) ** 6]
                ans_f1_2.append(ans_f1_temp)
    return ans_f1_2


def g_x_2(x):  # if x2...
    ans_g_x_2 = []
    for i in range(2*population_size):
        temp_2 = []
        for j in range(d):
            if j != 0:
                temp_2.append(x[i][j])
        # print(temp)
        ans_2 = 1 + 9 * (sum(temp_2) / (d - 1)) ** 0.25
        ans_g_x_2.append([ans_2])
    return ans_g_x_2


def f2_2(function1_values, g_x_values):
    ans_f2_2 = []
    for i in range(2*population_size):
        for j in range(1):
            ans_f2_temp = g_x_values[i][j] * \
                (1 - (function1_values[i][j] / g_x_values[i][j]) ** 2)
        ans_f2_2.append([ans_f2_temp])
    return ans_f2_2


'''-------- fast_non_dominated_sort --------'''


def non_dominated_sorting(population_size, fitness_values):
    s, n = {}, {}  # 初始化 S set, len = pop_size # 初始化被支配的次數
    front, rank = {}, {}  # 初始化 to store rank, level set
    front[0] = []  # 一開始為空集合
    for p in range(population_size*2):  # for each p<P
        s[p] = []  # 第0次為空值(P支配次數)
        n[p] = 0  # 第0次為空值(被支配的次數)
        for q in range(population_size*2):  # for each q<P
            # fitness_values[p][0] = value1, fitness_values[p][1] = value2
            if ((fitness_values[p][0] < fitness_values[q][0] and fitness_values[p][1] < fitness_values[q][1]) or (fitness_values[p][0] <= fitness_values[q][0] and fitness_values[p][1] < fitness_values[q][1])
                    or (fitness_values[p][0] < fitness_values[q][0] and fitness_values[p][1] <= fitness_values[q][1])):
                if q not in s[p]:  # if p 支配 q
                    s[p].append(q)  # add q to the set of p
            elif ((fitness_values[p][0] > fitness_values[q][0] and fitness_values[p][1] > fitness_values[q][1]) or (fitness_values[p][0] >= fitness_values[q][0] and fitness_values[p][1] > fitness_values[q][1])
                  or (fitness_values[p][0] > fitness_values[q][0] and fitness_values[p][1] >= fitness_values[q][1])):
                n[p] = n[p]+1  # increment the domination conter of p
        if n[p] == 0:  # p屬於first front
            rank[p] = 0  # rank 1
            if p not in front[0]:
                front[0].append(p)

    i = 0  # initial the front counter
    while (front[i] != []):
        Q = []  # to srore the members of the next front
        for p in front[i]:
            for q in s[p]:
                n[q] = n[q]-1
                if n[q] == 0:  # q belongs to next front
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)  # find rank 2, 3...
        i = i+1
        front[i] = Q

    del front[len(front)-1]
    return front


'''--------calculate crowding distance function---------'''


def calculate_crowding_distance(front, fitness_values):

    distance = {m: 0 for m in front}
    for o in range(2):
        obj = {m: fitness_values[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)  # 小到大排序
        distance[sorted_keys[0]
                 ] = distance[sorted_keys[len(front)-1]] = 999999999999  # 起,終點設inf
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
        for i in range(len(front)):  # non_dominated_sorting level:越高level的被優先選擇
            N = N+len(front[i])
            if N > population_size:  # calculate_crowding_distance:越大的被優先選擇
                distance = calculate_crowding_distance(
                    front[i], fitness_values)
                sorted_cdf = sorted(distance, key=distance.get)  # 小到大
                sorted_cdf.reverse()  # 大到小
                for j in sorted_cdf:
                    if len(new_pop) == population_size:
                        break
                    new_pop.append(j)  # 大到小
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
            for i in range(population_size)]    # 20*30
function1_values = f1(solution)
g_x_values = g_x(solution)
function2_values = f2(function1_values, g_x_values)

print('solution = init pop\n = ', solution)
print('solution_len\n = ', np.shape(solution))
print('function1_values\n = ', function1_values)
print('g_x_values\n = ', g_x_values)
print('function2_values\n = ', function2_values)
print('function2_values_len\n = ', np.shape(function2_values))

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
        alpha = 0.50
        parent_1 = solution[S[2*m]][:]  # temp parent
        parent_2 = solution[S[2*m+1]][:]
        child_1_temp = []
        child_2_temp = []
        for j in range(d):
            child_1 = alpha * parent_1[j] + (1 - alpha) * parent_2[j]
            child_2 = alpha * parent_2[j] + (1 - alpha) * parent_1[j]
            # append child chromosome to offspring list
            child_1_temp.append(child_1)
            child_2_temp.append(child_2)
        offspring_list.extend((child_1_temp, child_2_temp))
    print('parent_1\n = ', parent_1)
    print('parent_2\n = ', parent_2)
    print('child_1\n = ', child_1)
    print('child_2\n = ', child_2)
    print('child_1_temp\n = ', child_1_temp)
    print('child_2_temp\n = ', child_2_temp)
    print('offspring_list\n = ', offspring_list)
    print('offspring_list_len\n = ', np.shape(offspring_list))

    '''--------mutatuon--------'''
    Pm = 0.50
    number_of_time = []  # 要突變基因的位置
    mutationpopulation = np.copy(offspring_list)  # 先copy crossover後的染色體
    offspring_array = np.array(offspring_list)
    m, n = offspring_array.shape                # 20*10
    gene_number = m*n                           # 200
    # print('gene_number\n = ', gene_number)
    randoms = np.random.rand(gene_number)       # 隨機產生200個機率(0~1之間)
    for i in range(gene_number):
        if (randoms[i] < Pm):
            number_of_time.append(i)

    # 　print('變異基因位置:\n', number_of_time)
    for gene in number_of_time:
        gene_row = gene / n                     # 確定變異基因位於第幾條染色體
        gene_row = int(gene_row)
        gene_witch_col = gene % n               # 確定變異基因位於當前染色體的第幾個基因位
        gene_witch_col = int(gene_witch_col)
        # mutation
        if (mutationpopulation[gene_row, gene_witch_col] <= 0.5):
            mutationpopulation[gene_row, gene_witch_col] = 0
        elif (mutationpopulation[gene_row, gene_witch_col] > 0.5):
            mutationpopulation[gene_row, gene_witch_col] = 1

    mutationpopulation_list = mutationpopulation.tolist()
    print('mutationpopulation_list\n = ', mutationpopulation_list)

    '''--------fitness value-------------'''
    total_chromosome = copy.deepcopy(
        parent_list)+copy.deepcopy(mutationpopulation_list)  # combine parent and offspring chromosomes
    print('total_chromosome\n = ', total_chromosome)
    print('total_chromosome_len\n = ', np.shape(total_chromosome))  # 2N * d
    f1_2_ans = f1_2(total_chromosome)  # 2N * 1
    g_x_2_ans = g_x_2(total_chromosome)
    f2_2_ans = f2_2(f1_2_ans, g_x_2_ans)  # 2N * 1
    print('f1_2_ans\n = ', f1_2_ans)
    print('f2_2_ans\n = ', f2_2_ans)
    fitness_values = np.hstack((f1_2_ans, f2_2_ans))  # 2N * 2
    print('fitness_values\n = ', fitness_values)

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


final_function1_values = f1(population_list)
final_g_x_values = g_x(population_list)
final_function2_values = f2(final_function1_values, final_g_x_values)
final_fitness_values = np.hstack(
    (final_function1_values, final_function2_values))
print('Population\n = ',  np.array(population_list))
plt.scatter(final_fitness_values[:, 0], final_fitness_values[:, 1])
plt.xlabel('Objective function 1')
plt.ylabel('Objective function 2')
plt.legend(['pareto front'])
plt.title('MOGA of ZDT6')
plt.grid()
plt.show()
# sucess, x1 = [0, 1], x2...n = 0
