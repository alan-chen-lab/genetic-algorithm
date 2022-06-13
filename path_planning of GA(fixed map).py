import random
import numpy as np
import math
import matplotlib.pyplot as plt


class Fiter:
    def __init__(self):
        self.b = 1

    def function(self, a):
        for i in a:
            a = a[a.index(i) + 1:]
            if i in a:
                return i, 1
            else:
                pass
        return 0, 0

    def fiter(self, a):
        while (self.b == 1):
            (i, self.b) = self.function(a)
            c = [j for j, x in enumerate(a) if x == i]
            a = a[0:c[0]] + a[c[-1]:]
        return a


fiter = Fiter()  # init


# step1 初始化地圖
class Map():

    def __init__(self, row, col):

        self.data = []
        self.row = row  # 列
        self.col = col  # 行

    def map_init(self):  # map初始化，生成全為0矩陣

        self.data = [[0 for i in range(self.col)] for j in range(self.row)]

    def map_fix_Obstacle(self):    # 生成帶有障礙的map
        self.data[4][2] = 1
        self.data[4][3] = 1
        self.data[4][4] = 1
        self.data[4][5] = 1
        self.data[4][6] = 1
        self.data[5][2] = 1
        self.data[5][3] = 1
        self.data[5][4] = 1
        self.data[5][5] = 1
        self.data[5][6] = 1

        self.data[9][2] = 1
        self.data[9][3] = 1
        self.data[9][4] = 1
        self.data[9][5] = 1
        self.data[9][6] = 1
        self.data[10][2] = 1
        self.data[10][3] = 1
        self.data[10][4] = 1
        self.data[10][5] = 1
        self.data[10][6] = 1

        self.data[14][2] = 1
        self.data[14][3] = 1
        self.data[14][4] = 1
        self.data[14][5] = 1
        self.data[14][6] = 1
        self.data[15][2] = 1
        self.data[15][3] = 1
        self.data[15][4] = 1
        self.data[15][5] = 1
        self.data[15][6] = 1

        self.data[14][13] = 1
        self.data[14][14] = 1
        self.data[14][15] = 1
        self.data[14][16] = 1
        self.data[14][17] = 1
        self.data[15][13] = 1
        self.data[15][14] = 1
        self.data[15][15] = 1
        self.data[15][16] = 1
        self.data[15][17] = 1

        self.data[9][13] = 1
        self.data[9][14] = 1
        self.data[9][15] = 1
        self.data[9][16] = 1
        self.data[9][17] = 1
        self.data[10][13] = 1
        self.data[10][14] = 1
        self.data[10][15] = 1
        self.data[10][16] = 1
        self.data[10][17] = 1

        self.data[4][13] = 1
        self.data[4][14] = 1
        self.data[4][15] = 1
        self.data[4][16] = 1
        self.data[4][17] = 1
        self.data[5][13] = 1
        self.data[5][14] = 1
        self.data[5][15] = 1
        self.data[5][16] = 1
        self.data[5][17] = 1

        self.data[12][9] = 1
        self.data[12][10] = 1
        self.data[13][9] = 1
        self.data[13][10] = 1

        self.data[7][9] = 1
        self.data[7][10] = 1
        self.data[8][9] = 1
        self.data[8][10] = 1

        self.data[2][9] = 1
        self.data[2][10] = 1
        self.data[3][9] = 1
        self.data[3][10] = 1

        self.data[16][9] = 1
        self.data[16][10] = 1
        self.data[17][9] = 1
        self.data[17][10] = 1

        self.data[1][3] = 1
        self.data[2][3] = 1
        self.data[17][16] = 1
        self.data[18][16] = 1
        return self.data


# step2 初始化族群
class Population():
    def __init__(self, row, col, NP):
        self.row = row
        self.col = col
        self.NP = NP
        self.p_start = 0    # start point
        self.p_end = self.row * self.col - 1  # end point
        self.xs = (self.p_start) // (self.col)  # row
        self.ys = (self.p_start) % (self.col)  # col
        self.xe = (self.p_end) // (self.col)
        self.ye = (self.p_end) % (self.col)
        self.map_start = Map(self.row, self.col)  # init map
        self.map_start.map_init()
        self.map = self.map_start.map_fix_Obstacle()  # 生成帶有障礙的map
        plt.imshow(self.map, cmap='Greys')
        my_x_ticks = np.arange(0, 20, 1)
        my_y_ticks = np.arange(0, 20, 1)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)
        plt.title('Map')
        # plt.show()

        self.can = []  # 存不是障礙物的point，即可行走之點
        self.popu = [[0 for i in range(self.col)]
                     for j in range(self.NP)]  # 初始化族群
        # self.popu = []
        self.end_popu = []

    def Population_init_and_map(self):
        # 找到一條間斷但沒有障礙物的路徑，即從柵格圖的每一行中隨機取出一個不是障礙物的元素(但不連續)
        for i in range(self.NP):  # population number
            j = 0
            for xk in range(0, self.row):
                self.can = []  # 存可行走的點(0~24)
                for yk in range(0, self.col):
                    num = (yk) + (xk) * self.col  # 計算點 = x*row + y
                    if self.map_start.data[xk][yk] == 0:
                        self.can.append(num)
                # print(self.can)
                length = len(self.can)
                # print(length)
                self.popu[i][j] = (
                    self.can[random.randint(0, length-1)])  # 隨機挑選(0~24)，存到族群
                j += 1
            self.popu[i][0] = self.p_start  # 定義起點座標點
            self.popu[i][-1] = self.p_end  # 定義終點座標點

            temp = self.Generate_Continuous_Path(
                self.popu[i])
            if temp != []:
                temp = fiter.fiter(temp)  # 刪除一條路徑中重複之點
                self.end_popu.append(temp)
            # print(self.end_popu, end='\n')
        return self.end_popu

    def Generate_Continuous_Path(self, old_popu):  # 生成連續路径population

        self.new_popu = old_popu
        self.lengh = len(self.new_popu)
        i = 0
        # print("lengh =",self.lengh )
        while i != self.lengh-1:
            x_now = (self.new_popu[i]) // (self.col)    # row
            y_now = (self.new_popu[i]) % (self.col)  # col
            x_next = (self.new_popu[i+1]) // (self.col)  # next row
            y_next = (self.new_popu[i+1]) % (self.col)  # next col

            max_iteration = 0

            # 判斷下一個點與當前點的坐標是否連續(等於1，為連續)
            while max(abs(x_next - x_now), abs(y_next - y_now)) != 1:
                x_insert = math.ceil((x_next + x_now) // 2)
                y_insert = math.ceil((y_next + y_now) // 2)  # ceil向上取整数
                # print("x_insert = ",x_insert,"\n y_insert = ",y_insert)
                count = 0

                # 若插入點完0(可走)
                if self.map_start.data[x_insert][y_insert] == 0:
                    num_insert = (y_insert) + (x_insert) * \
                        self.col
                    self.new_popu.insert(i+1, num_insert)  # 插入得到之點
                    # print(self.new_popu)
                    # print(num_insert)
                else:  # 插入的柵格為障礙，判斷插入的柵格上下左右是否為障礙，以及是否在路徑中，若不是障礙且不在路徑中，就插入
                    # 判斷下方
                    if (x_insert + 1 < self.row) and count == 0:  # 保證坐標是在地圖上的
                        if ((self.map_start.data[x_insert+1][y_insert] == 0)  # 下方不是障礙物
                                and (((y_insert) + (x_insert+1) * self.col) not in self.new_popu)):  # 編號不在已知路徑中
                            num_insert = (y_insert) + \
                                (x_insert+1) * self.col  # 計算下方的編號
                            self.new_popu.insert(i + 1, num_insert)  # 插入編號
                            count = 1

                            # print('下方插入',num_insert)
                    # 判斷右方
                    if (y_insert + 1 < self.col) and count == 0:  # 保證坐標是在地圖上的
                        if ((self.map_start.data[x_insert][y_insert+1] == 0)  # 右方不是障碍物
                                and (((y_insert+1) + (x_insert) * self.col) not in self.new_popu)):  # 編號不在已知路徑中
                            num_insert = (y_insert+1) + \
                                (x_insert) * self.col  # 計算右方的編號
                            self.new_popu.insert(i + 1, num_insert)  # 插入編號
                            count = 1
                            # print('右方插入',num_insert)
                    # 判斷上方
                    if (x_insert - 1 > 0) and count == 0:  # 保證坐標是在地圖上的
                        if ((self.map_start.data[x_insert-1][y_insert] == 0)  # 右方不是障碍物
                                and (((y_insert) + (x_insert-1) * self.col) not in self.new_popu)):  # 編號不在已知路徑中
                            num_insert = (y_insert) + \
                                (x_insert-1) * self.col  # 計算上方的編號
                            self.new_popu.insert(i + 1, num_insert)  # 插入編號
                            count = 1
                            # print('上方插入',num_insert)
                    # 判斷左方
                    if (y_insert - 1 > 0) and count == 0:  # 保證坐標是在地圖上的
                        if ((self.map_start.data[x_insert][y_insert-1] == 0)  # 右方不是障碍物
                                and (((y_insert-1) + (x_insert) * self.col) not in self.new_popu)):  # 編號不在已知路徑中
                            num_insert = (y_insert-1) + \
                                (x_insert) * self.col  # 計算左方的編號
                            self.new_popu.insert(i + 1, num_insert)  # 插入編號
                            count = 1
                            # print('左方插入',num_insert)
                    if count == 0:  # 如果前面沒有插入新點，說明這條路徑不對，刪除
                        self.new_popu = []
                        break
                x_next = num_insert//self.col
                y_next = num_insert % self.col
                # x_next = x_insert
                # y_next = y_insert
                max_iteration += 1  # 迭代次数+1
                if max_iteration > 20:
                    self.new_popu = []
                    break
            if self.new_popu == []:
                break
            self.lengh = len(self.new_popu)
            i = i+1
        # print(self.new_popu)
        return self.new_popu


# step3 計算fitness value
def path_value(popu, col):
    value_path = []  # 存值
    for i in range(len(popu)):
        value_path.append(0)
        single_popu = popu[i]  # 一列列看(row)
        single_lengh = len(single_popu)
        for j in range(single_lengh-1):
            x_now = (single_popu[j]) // (col)
            y_now = (single_popu[j]) % (col)
            x_next = (single_popu[j + 1]) // (col)  # 路徑中下一個點座標
            y_next = (single_popu[j + 1]) % (col)
            if abs(x_now - x_next) + abs(y_now - y_next) == 1:  # 路徑為上下左右連續的 = 1
                value_path[i] = value_path[i] + 1
            elif max(abs(x_now - x_next), abs(y_now - y_next)) >= 2:  # 當穿過障礙物或跳太多格，逞罰
                value_path[i] = value_path[i] + 1000
            else:
                value_path[i] = value_path[i] + math.sqrt(2)  # 對角 = 根號2
    return value_path


def path_smooth(popu, col):  # add
    value_smooth = []
    for i in range(len(popu)):
        value_smooth.append(0)
        single_popu = popu[i]  # 一列列看(pop)
        single_lengh = len(single_popu)
        for j in range(single_lengh-2):
            x_now = (single_popu[j]) // (col)   # row
            y_now = (single_popu[j]) % (col)  # col
            # 點i+1所在之列行
            x_next1 = (single_popu[j + 1]) // (col)
            y_next1 = (single_popu[j + 1]) % (col)
            # 點i+2所在之列行
            x_next2 = (single_popu[j + 2]) // (col)
            y_next2 = (single_popu[j + 2]) % (col)
            # cos A = (b**2+c**2-a**2)/2bc
            b2 = (x_now - x_next1) ** 2 + (y_now - y_next1) ** 2
            c2 = (x_next2 - x_next1) ** 2 + (y_next2 - y_next1) ** 2
            a2 = (x_now - x_next2) ** 2 + (y_now - y_next2) ** 2
            cosa = (c2 + b2 - a2) / (2 * math.sqrt(b2) * math.sqrt(c2))
            angle = math.acos(cosa)  # radians
            angle_degree = math.degrees(angle)  # degree

            if (angle_degree < 170 and angle_degree > 91):
                value_smooth[i] = value_smooth[i] + 3
            elif (angle_degree <= 91 and angle_degree > 46):
                value_smooth[i] = value_smooth[i] + 25
            elif (angle_degree <= 46):
                value_smooth[i] = value_smooth[i] + 50
    return value_smooth


# step4 selection
def selection(pop, value_path, value_smooth, a, b):
    now_value = []  # fitness value list
    now_value2 = []  # value_smooth
    P_value = []  # cumsum
    random_number = []  # random_number
    new_popu = []  # after select
    all_fitness = []  # fitness value list + value_smooth
    total_value = 0
    sum_value = 0
    sum_value2 = 0  # value_smooth
    sum_all = 0  # value_all
    lengh = len(pop)
    for i in range(lengh):
        new_popu.append([])
    # -------------------------------------------------value_path
    for i in value_path:
        now_value.append(1/i)  # 取倒數，距離越短，值越大
        sum_value += (1/i)  # sum of fitness
    # -------------------------------------------------value_smooth
    for i in value_smooth:
        now_value2.append(1/i)   # 取倒數，角度越大，值越大
        sum_value2 += (1/i)  # sum of fitness
    # -------------------------------------------------value_path + value_smooth
    for i in range(len(now_value)):
        total_value = a * now_value[i] + b * \
            now_value2[i]  # fit = a * path + b * smooth
        all_fitness.append(total_value)

    # 輪盤法
    for i in all_fitness:  # 加總
        sum_all += (i)
    for i in all_fitness:
        P_value.append(i/sum_all)  # 算每個fitness value的比率
    P_value = np.cumsum(P_value)  # 累加

    for i in range(lengh):
        random_number.append(random.random())  # random_number
    random_number = sorted(random_number)  # 小到大
    p_index = 0
    new_index = 0
    while(new_index < lengh):
        if random_number[new_index] < P_value[p_index]:  # Q1<randoms<Q2 --> V2
            new_popu[new_index] = pop[p_index]
            new_index += 1
        else:
            p_index += 1

    # 使用Tournament Selection(k = 2)
    return new_popu, all_fitness


# step5 crossover
def crossover(parents, pc):
    children = []
    lenparents = len(parents)  # cross前population長度
    parity = lenparents % 2  # 若為基數個，不cross
    for i in range(0, lenparents-1, 2):
        single_now_popu = parents[i]
        single_next_popu = parents[i+1]
        children.append([])
        children.append([])
        index_content = list(set(single_now_popu).intersection(
            set(single_next_popu)))  # 第一條與第二條路徑重複部分
        num_rep = len(index_content)  # 重複內容個數
        if random.random() < pc and num_rep >= 3:   # 要change
            content = index_content[random.randint(
                0, num_rep-1)]  # 隨機選一個重複內容
            now_index = single_now_popu.index(content)  # 重複內容在哪個位置
            next_index = single_next_popu.index(content)  # 重複內容在哪個位置
            children[i] = single_now_popu[0:now_index + 1] + \
                single_next_popu[next_index + 1:]
            children[i+1] = single_next_popu[0:next_index + 1] + \
                single_now_popu[now_index + 1:]
        else:
            children[i] = parents[i]
            children[i+1] = parents[i+1]
    if parity == 1:  # 如果是基數，加一條空的
        children.append([])
        children[-1] = parents[-1]
    return children


# step6 mutation
def mutation(children, pm):
    row = len(children)
    new_popu = []
    for i in range(row):
        temp = []
        while(temp == []):  # 等待mutation
            single_popu = children[i]
            if random.random() < pm:  # <pm，mutation
                col = len(single_popu)
                first = random.randint(1, col-2)  # 隨機挑兩個點
                second = random.randint(1, col-2)
                if first != second:  # 判斷兩個點是否相同
                    if(first < second):
                        single_popu = single_popu[0:first] + \
                            single_popu[second+1:]
                    else:
                        single_popu = single_popu[0:second] + \
                            single_popu[first+1:]
                temp = population.Generate_Continuous_Path(single_popu)  # 連續化
                if temp != []:
                    new_popu.append(temp)

            else:  # 不mutation直接賦值
                new_popu.append(single_popu)
    return new_popu


if __name__ == '__main__':
    row = 20
    col = 20
    pop = 40
    a = 5   # adjust
    b = 2
    print('原始地圖:')
    # 族群初始化，且初始化地圖(row, col, obstacle_number, pop)
    population = Population(row, col, pop)

    popu = population.Population_init_and_map()  # init population

    generation = []
    best_distance_value = []
    mean_distabce_value = []
    fitness_val = []
    iteration = 30
    for i in range(iteration):
        print(i)
        lastpopu = popu  # 保存上一次族群
        value = path_value(popu, population.col)  # 計算distance

        value_smooth = path_smooth(popu, population.col)  # 計算path smooth

        after_selection, fitness_values = selection(
            popu, value, value_smooth, a, b)  # selection

        child = crossover(after_selection, 0.50)  # crossover

        popu = mutation(child, 0.80)  # mutation

        if popu == []:
            popu = lastpopu
            break
        generation.append(i)
        fitness_val.append(np.max(fitness_values))
        best_distance_value.append(np.min(value))  # 計算最小distance value
        mean_distabce_value.append(np.mean(value))  # 計算平均distance value

    if popu == []:
        print('no path!')
    else:
        value = path_value(popu, population.col)  # 計算fitness value
        minnum = value[0]
        for i in range(len(value)):  # 找最小之fitness value
            if value[i] < minnum:
                minnum = value[i]
        popu = popu[value.index(minnum)]

        path_record_x = []
        path_record_y = []
        for i in popu:
            x = (i) // (population.map_start.col)  # 列
            y = (i) % (population.map_start.col)  # 行
            path_record_x.append(y)
            path_record_y.append(x)
            population.map_start.data[x][y] = '*'

        plt.plot(path_record_x[0], path_record_y[0], 'sr')    # start point
        plt.plot(path_record_x[-1], path_record_y[-1], 'sg')    # end point
        plt.plot(path_record_x, path_record_y, '-b')    # after planning
        plt.legend(['start', 'end', 'path'])
        plt.grid(True)
        plt.show()
        print('\n路徑規劃後之地圖:')
        for i in range(population.map_start.row):
            for j in range(population.map_start.col):
                print(population.map_start.data[i][j], end=' ')
            print('')

        print('最短路徑值：', minnum)
        print('最短路徑之族群：', popu)
        plt.plot(generation, best_distance_value, "-r", linewidth=3.0)
        plt.plot(generation, mean_distabce_value, "-b")
        # plt.text(125, 200, 'The shortest path value = %.4f' %
        #          (minnum), fontsize=10)
        plt.title('Path planning of GA')
        plt.ylabel("distance")  # y label
        plt.xlabel("generation")  # x label
        plt.legend(
            ['best distance value', 'mean distance value'])
        plt.grid(True)
        plt.show()

        plt.plot(generation, fitness_val, "-b")
        plt.title('Path planning of GA')
        plt.ylabel("fitness value")  # y label
        plt.xlabel("generation")  # x
        plt.legend(['fitness value'])
        plt.grid(True)
        plt.show()
