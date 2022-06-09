import numpy as np
import random
import math
import matplotlib.pyplot as plt


def initial_array(populationSize, d):
    initial_array = []
    for i in range(populationSize):
        for j in range(d):
            initial_value = random.randint(-600, 600)
            initial_array.append(initial_value)
    initial_pop = np.reshape(
        initial_array, (populationSize, d)).astype(np.float32)  # 初始化族群
    return initial_pop


def multimodel_function(initial_pop):   # 題目(mutimodel)
    pop, d = initial_pop.shape
    temp = np.zeros((pop, d), dtype=np.float32)
    total = np.zeros((pop, 1), dtype=np.float32)
    for a in range(1, pop+1):   # 1~20
        for b in range(1, d + 1):
            temp[a-1, b-1] = initial_pop[a-1, b-1] / \
                float(math.sqrt(b))  # update

        # temp[a-1, :] = initial_pop[a-1, :]/float(math.sqrt(a))
        total[a-1, :] = (1/4000) * sum(initial_pop[a-1, :] **
                                       2) - np.prod(np.cos(temp[a-1, :])) + 1
    return total


population_size = 1
d = 30
iteration = 2000  # Markov_chain length (inner loops numbers)
counter = 1
# alpha = 0.9  # 衰減參數
alpha = 1/(np.log(counter) + 1)  # Boltzmann
T = 10000  # 初始T
T_min = 1e-20  # 最小T
iter = 0
record_iter = []
fitness_val = []
generation_time = []
# --------------設定初始值
init_x = initial_array(population_size, d)  # init
new_x = init_x.copy()

while T > T_min:
    for i in range(iteration):

        x_value = multimodel_function(init_x)  # fitness value

        for j in range(d):
            new_x[:, j] = init_x[:, j] + T * np.random.uniform(-1000, 1000)

            # 不能超過上下限
            if new_x[:, j] > 600:
                new_x[:, j] = 600
            elif new_x[:, j] < -600:
                new_x[:, j] = -600

        new_x_value = multimodel_function(new_x)  # new fitness value

        # Metropolis
        for a in range(population_size):
            if (new_x_value[a] - x_value[a] < 0):  # new - x < 0
                # 接受新解
                init_x = new_x.copy()
            else:  # new - x > 0
                temp = -(new_x_value[a] - x_value[a]) / T
                P = np.exp(temp)

                if P > np.random.uniform(0, 1):
                    init_x = new_x.copy()

    a = multimodel_function(init_x)
    for q in range(1):
        temp = a[q]
    for z in range(1):
        temp1 = temp[z]
    fitness_val.append(temp1.tolist())
    record_iter.append(iter)

    T = T * (1/(np.log(counter) + 1))  # 降溫
    counter = counter + 1
    iter = iter + 1
    # T = alpha * T  # 降溫
    print('T =', T)
    print('fitness =', multimodel_function(init_x))

fitness = np.min(fitness_val)
print("best fitness (accept error <= 0.01) = \n", fitness)
plt.title('multimodel')
plt.xlabel('temperature iter')
plt.ylabel('fitness value')
plt.plot(record_iter, fitness_val, 'b-', label='best fitness value')
plt.legend()
plt.grid(True)
plt.show()
