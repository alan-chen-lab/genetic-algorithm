import numpy as np
import random
import copy
from scipy.optimize import fsolve, basinhopping
import timeit
import matplotlib.pyplot as plt

# 得到染色體長度
def EncodedLength(decimal_places=0.0001, boundarylist=[]):
    lengths = []                        # 染色體長度
    for i in boundarylist:
        lower = i[0]
        upper = i[1]
        # 50代表可容納之長度(m)
        m = fsolve(lambda x: ((upper - lower) * 1 / decimal_places) - 2 ** x - 1, 50)
        # print(m) = 17.xxx, 14.xxx
        length = int(np.floor(m[0])+1)  # 18, 15
        lengths.append(length)          # 將得到的值插入到最後一格
        # print(lengths) = [18, 15]
    return lengths

# 隨機生成初始染色體族群(20)
def IntialPopulation(lengthEncode, populationSize):
    # 隨機化初始族群為0
    # 初始化成20列, 33行
    initial_array = np.zeros((populationSize, sum(lengthEncode)), dtype=np.uint8)
    # 0~19次
    for i in range(populationSize):
        # 將每個基因群定義為0 or 1
        initial_array[i, :] = np.random.randint(0, 2, sum(lengthEncode))
    return initial_array

# decode
def population_decoded(lengthEncode, initial_array, boundarylist):
    populations = initial_array.shape[0]    # 20
    # print(populations)
    variables = len(lengthEncode)           # 2
    # print(variables)
    decodedvalues = np.zeros((populations, variables))  # array(20*2)
    for j, array_to_list in enumerate(initial_array):
        array_to_list = array_to_list.tolist()
        start = 0
        for index, length in enumerate(lengthEncode):
            # print(index, length) --> index = 0,1 length = 18,15
            # 將基因進行拆分
            power = length - 1
            demical = 0
            for i in range(start, length + start):
                demical += array_to_list[i] * (2 ** power)
                power -= 1
            lower = boundarylist[index][0]
            upper = boundarylist[index][1]
            decodedvalue = lower + demical * (upper - lower) / (2 ** length - 1)
            decodedvalues[j, index] = decodedvalue   # x1, x2
            # 下一段基因decode
            start = length
    return decodedvalues

# illustrated example (function)
def illustrated_example():
    return lambda x: 21.5 + x[0] * np.sin(4 * np.pi * x[0]) + x[1] * np.sin(20 * np.pi * x[1])
    pass

# 將經過方程式計算後的答案(適應值)連加， 和累積權重
def summation(func, decoded):
    population, nums = decoded.shape           # 20*2
    # 初始化種群的適應值為0
    fitnessvalues = np.zeros((population, 1))  # 20*1
    # 連加適應值
    for i in range(population):
        fitnessvalues[i, 0] = func(decoded[i, :])
    # 計算每個基因被選擇的機率
    p = fitnessvalues / np.sum(fitnessvalues)
    total_fitness = np.sum(fitnessvalues)      # total_fitness
    # print('total_fitness:', np.sum(fitnessvalues))
    # 累加每個基因權重
    cumulative_p = np.cumsum(p)
    return total_fitness, fitnessvalues, cumulative_p

# 輪盤和產生新族群
def roulette(initial_encode, cum_p):
    m, n = initial_encode.shape                       # 20*33
    newpopulation = np.zeros((m, n), dtype=np.uint8)  # 初始化新族群
    randoms = np.random.rand(m)                       # 隨機產生20個機率(0~1之間)
    for j in range(m):
        for i in range(m-1):
            if(cum_p[i]<randoms[j]<cum_p[i+1]):       # Q10<randoms<Q11 --> V11
                newpopulation[j] = initial_encode[i+1]
    return randoms, newpopulation

# crossover基因
def crossover(population, Pc=0.25):
    m, n = population.shape  # 20*33
    # 新種群初始化
    temp_updatepopulation = np.zeros((m, n), dtype=np.uint8)  # 初始化要換的基因
    updatepopulation = np.zeros((m, n), dtype=np.uint8)
    no_need_crossover_randoms = []
    number_of_time = []                       # 需要cross的次數

    randoms = np.random.rand(m)               # 隨機產生20個機率(0~1之間)
    for i in range(m):
        if(randoms[i] > Pc):                  # 若隨機機率大於0.25
            no_need_crossover_randoms.append(randoms[i])
        else:                                                 # 若隨機機率小於0.25
            temp_updatepopulation[i, :] = population[i, :]    # 保留第i列，取所有行(要換的基因)
            number_of_time.append(i)                          # 存需要cross的次數
            ##　print('第', number_of_time[-1], '次機率小於Pc(要換)')

        if not number_of_time.__contains__(i):                # 不進行交叉的先copy
            updatepopulation[i, :] = population[i, :]
            ##　print('第', i, '次')

    # crossover
    while len(number_of_time) > 0:
        ##　print('need to crossover:', number_of_time)           # 確定與要cross的次數一樣
        a = number_of_time.pop()
        try:                                                      # 若剩下一項不夠pop()，則跳出
            b = number_of_time.pop()
        except:
            updatepopulation[a, :] = population[a, :]             # 複製剩下之基數項的基因
            break
        ##　print('pop:', b, a)

        # 隨機產生一個交叉點
        crossoverPoint = random.sample(range(1, n), 1)
        crossoverPoint = crossoverPoint[0]

        updatepopulation[a, 0:crossoverPoint] = population[a, 0:crossoverPoint]     # 換後面
        updatepopulation[a, crossoverPoint:] = population[b, crossoverPoint:]       # b-->a

        updatepopulation[b, 0:crossoverPoint] = population[b, 0:crossoverPoint]     # 換後面
        updatepopulation[b, crossoverPoint:] = population[a, crossoverPoint:]       # a-->b

    return updatepopulation

# 染色體變異
def mutation(population, Pm=0.01):
    number_of_time = []                         # 要突變基因的位置
    mutationpopulation = np.copy(population)    # 先copy crossover後的染色體
    # mutationpopulation = copy.deepcopy(population) 
    m, n = population.shape                     # 20*33
    gene_number = m*n                           # 660
    randoms = np.random.rand(gene_number)       # 隨機產生660個機率(0~1之間)
    for i in range(gene_number):
        if (randoms[i] < Pm):
            number_of_time.append(i)

    ##　print('變異基因位置:\n', number_of_time)
    for gene in number_of_time:
        gene_row = gene / n                     # 確定變異基因位於第幾條染色體(n=33)
        gene_row = int(gene_row)
        gene_witch_col = gene % n               # 確定變異基因位於當前染色體的第幾個基因位
        gene_witch_col = int(gene_witch_col)
        ##　print('row, col:[', gene_row, gene_witch_col, ']')
        # mutation
        if (mutationpopulation[gene_row, gene_witch_col] == 0):  # 若第幾列第幾行為0，則變1
            mutationpopulation[gene_row, gene_witch_col] = 1
        else:
            mutationpopulation[gene_row, gene_witch_col] = 0

    return mutationpopulation


# start
def main(iteration=1000):
    optimalValues = []
    optimalSolutions = []
    generation_time = []
    max_generation_time = []
    iteration_record = []
    optimal_v = []
    search_space = [[-3.0, 12.1], [4.1, 5.8]]
    lengthEncode = EncodedLength(boundarylist=search_space)     # 得到初始染色體長度
    initial_Population = IntialPopulation(lengthEncode, 20)     # 得到初始種群基因
    # print('initial_encode:\n', initial_encode)
    # print('\n')
    # print(lengthEncode)
    for iteration in range(iteration):                          # iteration次數
        # decode
        decoded = population_decoded(lengthEncode, initial_Population, search_space)
        # print(decoded[0])
        # 將經過方程式計算後的答案得到的evaluate，以及除以連加後的cumulative
        total_fitness, evaluate, cum_p = summation(illustrated_example(), decoded)
        ## print('total_fitness:\n', total_fitness)
        # print('cum_p:\n', cum_p)
        # print('\n')
        # 輪盤產生之機率以及新的族群
        randoms_p, newpopulation = roulette(initial_Population, cum_p)
        # print('randoms_p:\n', randoms_p)
        # print('\n')
        ## print('newpopulation:\n', newpopulation)

        # crossover後得到新的族群
        updatepopulation = crossover(newpopulation)
        ##　print('crossoverpopulation:\n', updatepopulation)

        # 基因變異後結果
        mutationpopulation = mutation(updatepopulation)
        ## print('mutationpopulation:\n', mutationpopulation)

        # mutation取代initial_encode
        initial_Population = mutationpopulation  

        # 將變異後的族群decode，得到每輪迭代最終的族群
        final_decoded = population_decoded(lengthEncode, mutationpopulation, search_space)
        ## print('final_decoded:\n', final_decoded)

        # 變異後的族群的適應度evaluate
        after_crossover_total_fitness, after_crossover_evaluate, after_crossover_cum_p = summation(illustrated_example(), final_decoded)
        ##　print('after_crossover_total_fitness:\n', after_crossover_total_fitness)
        ## print('after_crossover_evaluate:\n', after_crossover_evaluate)
        ##　print('after_crossover_cum_p:\n', after_crossover_cum_p)

        # 紀錄每一次迭代的最優適應度值
        optimalValues.append(np.max(list(after_crossover_evaluate)))     
        ##　print('optimal_fitness_value:\n', optimalValues)
        # 紀錄最優適應度值的x1,x2
        index = np.where(after_crossover_evaluate == max(list(after_crossover_evaluate)))
        optimalSolutions.append(final_decoded[index[0][0], :])
        ##　print('optimal_fitness_value_of_x1,x2:[', optimalSolutions[0][0], optimalSolutions[0][1], ']\n')
        generation_time.append(iteration)

        # 每迭代100次紀錄一次並且記下目前最優的適應度值
        if(iteration == 99):
            iteration_record.append(iteration)
            value_100 = np.max(optimalValues)
            optimal_v.append(value_100)
        elif(iteration == 199):
            iteration_record.append(iteration)
            value_200 = np.max(optimalValues)
            optimal_v.append(value_200)
        elif(iteration == 299):
            iteration_record.append(iteration)
            value_300 = np.max(optimalValues)
            optimal_v.append(value_300)
        elif(iteration == 399):
            iteration_record.append(iteration)
            value_400 = np.max(optimalValues)
            optimal_v.append(value_400)
        elif(iteration == 499):
            iteration_record.append(iteration)
            value_500 = np.max(optimalValues)
            optimal_v.append(value_500)
        elif(iteration == 599):
            iteration_record.append(iteration)
            value_600 = np.max(optimalValues)
            optimal_v.append(value_600)
        elif(iteration == 699):
            iteration_record.append(iteration)
            value_700 = np.max(optimalValues)
            optimal_v.append(value_700)
        elif(iteration == 799):
            iteration_record.append(iteration)
            value_800 = np.max(optimalValues)
            optimal_v.append(value_800)
        elif(iteration == 899):
            iteration_record.append(iteration)
            value_900 = np.max(optimalValues)
            optimal_v.append(value_900)
        elif(iteration == 999):
            iteration_record.append(iteration)
            value_1000 = np.max(optimalValues)
            optimal_v.append(value_1000)
        # 假設迭代2000次
        # if(iteration == 1099):
        #     iteration_record.append(iteration)
        #     value_1099 = np.max(optimalValues)
        #     optimal_v.append(value_1099)
        # elif(iteration == 1199):
        #     iteration_record.append(iteration)
        #     value_1199 = np.max(optimalValues)
        #     optimal_v.append(value_1199)
        # elif(iteration == 1299):
        #     iteration_record.append(iteration)
        #     value_1299 = np.max(optimalValues)
        #     optimal_v.append(value_1299)
        # elif(iteration == 1399):
        #     iteration_record.append(iteration)
        #     value_1399 = np.max(optimalValues)
        #     optimal_v.append(value_1399)
        # elif(iteration == 1499):
        #     iteration_record.append(iteration)
        #     value_1499 = np.max(optimalValues)
        #     optimal_v.append(value_1499)
        # elif(iteration == 1599):
        #     iteration_record.append(iteration)
        #     value_1599 = np.max(optimalValues)
        #     optimal_v.append(value_1599)
        # elif(iteration == 1699):
        #     iteration_record.append(iteration)
        #     value_1699 = np.max(optimalValues)
        #     optimal_v.append(value_1699)
        # elif(iteration == 1799):
        #     iteration_record.append(iteration)
        #     value_1799 = np.max(optimalValues)
        #     optimal_v.append(value_1799)
        # elif(iteration == 1899):
        #     iteration_record.append(iteration)
        #     value_1899 = np.max(optimalValues)
        #     optimal_v.append(value_1899)
        # elif(iteration == 1999):
        #     iteration_record.append(iteration)
        #     value_1999 = np.max(optimalValues)
        #     optimal_v.append(value_1999)

    optimalValue = np.max(optimalValues)                            # 迭代後最優適應度值
    for i in range(iteration):
        a = optimalValues[i]
        if(optimalValue == a):
            max_generation_time.append(i)

    optimalIndex = np.where(optimalValues == optimalValue)          # 當最大的值有在每代最優解中
    optimalSolution = optimalSolutions[optimalIndex[0][0]]          # 迭代後最優適應度值的x1,x2

    return optimalSolution, optimalValue, generation_time, optimalValues, optimalSolutions, max_generation_time, iteration_record, optimal_v

best_solution, best_fitness_value, generation_time, optimalValues, optimalSolutions , max_generation , iteration_count, optimal_val= main()
print('計算出最優fitness value的x1: {:.4f}'.format(best_solution[0]))
print('\n')
print('計算出最優fitness value的x2: {:.4f}'.format(best_solution[1]))
print('\n')
print('最優fitness value: {:.4f} '.format(best_fitness_value))
print('\n')
print('在迭代第幾代有最優fitness value:', max_generation[0])
print('\n')
# 測量執行時間
time = timeit.timeit(stmt=main, number=1)
print('計算時間: {:.4f} '.format(time))

# print(len(generation_time))
# print(len(optimalValues))
# print(len(optimalSolutions))
# 2d plot
plt.subplot(2, 1, 1)
plt.plot(generation_time, optimalValues)
plt.ylabel("fitness values") # y label
plt.xlabel("generation") # x label

# print('iteration_count', iteration_count)
# print('optimal_val', optimal_val)
plt.subplot(2, 1, 2)
plt.plot(iteration_count, optimal_val)
plt.ylabel("fitness values") # y label
plt.xlabel("generation") # x label
plt.show()

# 3d plot all of x1, x2, number of generation
fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection ="3d")
optimalSolutions = np.array(optimalSolutions)
ax1.scatter3D(optimalSolutions[:,0], optimalSolutions[:,1], generation_time, c='r')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('generation')

ax2 = fig.add_subplot(1, 2, 2, projection ="3d")
ax2.scatter3D(best_solution[0], best_solution[1], max_generation[0], c='b')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('generation')
plt.show()
