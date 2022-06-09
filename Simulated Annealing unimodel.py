import numpy as np
import random
import matplotlib.pyplot as plt


def initial_array(populationSize, d):
    initial_array = []
    for i in range(populationSize):
        for j in range(d):
            initial_value = random.randint(-100, 100)
            initial_array.append(initial_value)
    initial_pop = np.reshape(
        initial_array, (populationSize, d)).astype('f')  # 初始化族群
    return initial_pop


def unimodel_function(initial_pop):
    pop, d = initial_pop.shape
    total = np.zeros((pop, 1), dtype=np.float32)
    # temp_init = np.zeros((pop, d), dtype=np.float32)  # update
    for a in range(pop):
        # temp_init = (np.cumsum(initial_pop, axis=1)
        #              ) ** 2   # [1,2,3] -> [1,3,6]

        # total[a, :] = sum(temp_init[a, :])    # 題目(unimodel), update
        total[a, :] = sum(np.cumsum(initial_pop[a, :]) **
                          2)    # 題目(unimodel), orginal
    # print("fitness value:\n", total)
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

        x_value = unimodel_function(init_x)  # fitness value

        for j in range(d):
            new_x[:, j] = init_x[:, j] + T * np.random.uniform(-0.1, 0.1)

            # 不能超過上下限
            if new_x[:, j] > 100:
                new_x[:, j] = 100
            elif new_x[:, j] < -100:
                new_x[:, j] = -100

        new_x_value = unimodel_function(new_x)  # new fitness value

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

    a = unimodel_function(init_x)
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
    print('fitness =', unimodel_function(init_x))

fitness = np.min(fitness_val)
print("best fitness (accept error <= 100) = \n", fitness)
plt.title('unimodel')
plt.xlabel('temperature iter')
plt.ylabel('fitness value')
plt.plot(record_iter, fitness_val, 'b-', label='best fitness value')
plt.legend()
plt.grid(True)
plt.show()
