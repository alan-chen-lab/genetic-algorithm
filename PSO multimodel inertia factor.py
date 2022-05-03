import numpy as np
import math as ms
import matplotlib.pyplot as plt
import random as rd


def mutimodel(x):   # 題目(unimodal)
    for b in range(1, dimension + 1):
        temp = x[b-1] / float(ms.sqrt(b))
        ans = (1/4000) * sum(x ** 2) - np.prod(np.cos(temp)) + 1

    return ans


def initial_X_V_pbest(pop, d):  # initial X,V,Pbest,Gbest
    position_X = np.zeros((pop, d))
    velocity = np.zeros((pop, d))
    Pbest = np.zeros((pop, d))
    for i in range(pop):  # row
        for j in range(d):  # col
            x = rd.uniform(search_space[0], search_space[1])
            # 速度極限 +-(下限-上限)/2, 初始給予亂數0~1增加收斂速度
            v = rd.uniform(0, 1)
            position_X[i][j] = x
            velocity[i][j] = v
    Pbest = position_X.copy()  # init pbest = position_X
    return position_X, velocity, Pbest


def find_pbest(x, copyX):  # pbest的x

    if mutimodel(x) < mutimodel(copyX):
        copyX = x
    else:
        copyX = copyX

    return copyX


def find_gbest(PPbest):  # find gbest

    Gbest = PPbest[0]   # if Gbest = PPbest[0]
    # print('Gbest', Gbest)
    for j in range(particles_number):  # row
        if j+1 == particles_number:    # j = 29, n = 30
            break
        else:

            if mutimodel(Gbest) > mutimodel(PPbest[j+1]):
                Gbest = PPbest[j+1]
            else:
                Gbest = Gbest

    return Gbest


def updated_X_V(x, v, pbest, gbest):  # inertia factor
    V_1 = w * v + (cognition_factor * (rd.random()) * (pbest-x)) + \
        (social_factor * (rd.random()) * (gbest-x))

    X_1 = x + V_1

    return (X_1, V_1)


particles_number = 50  # particles number(20~50)
dimension = 30  # dimension
generation = 10000
search_space = [-600, 600]
cognition_factor = 3.1
social_factor = 1.0  # lead to more local minima trapping

X, V, Pbest = initial_X_V_pbest(particles_number, dimension)    # x, v, pbest
Gbest = find_gbest(Pbest).copy()
copy_X = X.copy()
copy_V = V.copy()


if __name__ == "__main__":
    Gbest_y = []
    iter = 1
    # while stopping condition is not true do
    while iter <= generation:

        w = 0.9 - float(0.9 - 0.4)/generation * \
            iter   # for weight, more stable
        # for each particle do
        for i in range(particles_number):  # row
            # Update the particle’s velocity
            # Update the particle’s position
            copy_X[i], copy_V[i] = updated_X_V(
                X[i], V[i], Pbest[i], Gbest)  # get new x, v

            for j in range(dimension):  # col
                if copy_X[i][j] > search_space[1]:
                    copy_X[i][j] = search_space[1]
                elif copy_X[i][j] < search_space[0]:
                    copy_X[i][j] = search_space[0]
                if copy_V[i][j] > search_space[1]:
                    copy_V[i][j] = search_space[1]
                elif copy_V[i][j] < search_space[0]:
                    copy_V[i][j] = search_space[0]

            # Evaluate the fitness of the particle
            # Update the particle’s personal best position
            Pbest[i] = find_pbest(X[i], copy_X[i])
            if (Pbest[i] == X[i]).any == True:
                copy_X[i] = X[i].copy()
                copy_V[i] = V[i].copy()

        X = copy_X.copy()
        V = copy_V.copy()
        # Update the global best position
        Gbest = find_gbest(Pbest)

        ans = mutimodel(Gbest)  # final gbest
        Gbest_y.append(ans)
        print(iter)  # number
        print("Gbest=", ans)
        iter = iter + 1   # 1~1000

    G = mutimodel(Gbest)
    print("final Gbest (accept error<0.1)=", G)
    x = np.arange(0, generation)
    y = Gbest_y
    plt.plot(x, y)
    plt.title('PSO multimodel')
    plt.xlabel("generation")
    plt.ylabel("fitness values")
    plt.show()
