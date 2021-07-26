import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5 as pickle

# open the 'pkl' file you want to read
pkl_file = open('/Users/liyurun/code/Python/DataAnalysis_Seino/data/distance_20200613/distance_data_20200613.pkl', 'rb')

# read the data
data = pickle.load(pkl_file)
pkl_file.close()

POSITION_MATRIX = data[0][0][3]
POSITION = data[0][0][2]

N_CITIES = POSITION.shape[0]  # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.02
POP_SIZE = 100
N_GENERATIONS = 50


def cal_dis(lat1, lon1, lat2, lon2):
    lat1 = (math.pi / 180) * lat1
    lat2 = (math.pi / 180) * lat2
    lon1 = (math.pi / 180) * lon1
    lon2 = (math.pi / 180) * lon2

    R = 6378.137
    if (lat1 != lat2) & (lon1 != lon2):
        d = math.acos(
            math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * R * 1000
    else:
        d = 0
    return d


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, position):
        self.DNA_size = DNA_size  # locations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.position = position

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, position):  # get locations' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            position_coord = position[d]
            line_x[i, :] = position_coord[:, 0]
            line_y[i, :] = position_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, DNA):  # self.pop --> DNA
        total_distance = np.zeros(DNA.shape[0], dtype=int)
        for n in range(DNA.shape[0]):
            for loc in range(self.DNA_size - 1):  # number of routes
                total_distance[n] += cal_dis(self.position[DNA[n][loc]][0], self.position[DNA[n][loc]][1],
                                             self.position[DNA[n][loc + 1]][0], self.position[DNA[n][loc + 1]][1])
            total_distance[n] += cal_dis(self.position[DNA[n][0]][0], self.position[DNA[n][0]][1],
                                         self.position[DNA[n][-1]][0], self.position[DNA[n][-1]][1])
        fitness = 100000 / total_distance
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # choose crossover points
            keep_position = parent[~cross_points]  # find the city number
            swap_position = pop[i_, np.isin(pop[i_].ravel(), keep_position, invert=True)]
            parent[:] = np.concatenate((keep_position, swap_position))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
            return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, position):
        self.position = position
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.position[:, 0].T, self.position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-', color='red')
        seline_x = np.array([lx[0], lx[-1]])
        seline_y = np.array([ly[0], ly[-1]])
        plt.plot(seline_x.T, seline_y.T, 'r-', color='red')
        plt.title("Total distance=%f" % total_d, fontdict={'size': 14, 'color': 'red'})
        plt.xlim((min(self.position[:, 0]), max(self.position[:, 0])))
        plt.ylim((min(self.position[:, 1]), max(self.position[:, 1])))
        # plt.pause(0.01)


# def main():
#    for num in range(len(data)):
#        for times in range(len(data[num])):   #input position matrix
dist = [0 for i in range(len(data))]
ctime = [0 for i in range(len(data))]


def GA_run(num, times, cross_rate, mutate_rate):
    start = time.time()
    POSITION_MATRIX = data[num][times][3]
    POSITION = data[num][times][2]
    # set the GA parameter
    N_CITIES = POSITION.shape[0]  # DNA size
    CROSS_RATE = cross_rate
    MUTATE_RATE = mutate_rate
    POP_SIZE = int(2 * POSITION.shape[0] ** 3)
    N_GENERATIONS = int(10 * POSITION.shape[0])

    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE,
            position=POSITION)
    env = TravelSalesPerson(position=POSITION)
    shortest = 1000000
    best_path = ga.pop[0]
    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.position)
        fitness, total_distance = ga.get_fitness(ga.pop)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        if total_distance[best_idx] < shortest:
            shortest = total_distance[best_idx]
            best_path = ga.pop[best_idx]
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],
              '| total distance of this delivery: %f' % total_distance[best_idx], )
        env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

        print('===============================')
        print('the shortest distance of this delivery is: %2f\n' % shortest)

    plt.ioff()
    plt.show()

    #    data[num][times].append(shortest)
    #    data[num][times].append(best_path)
    #    print(data[num][times][4])
    #    print(data[num][times][5])
    end = time.time()
    dist[num] = shortest
    ctime[num] = end - start
    print('computation time : %f' % (end - start))
    print('driver ID : %d' % data[num][0][0])
    print('shortest :%f' % shortest)
    print(best_path)
    plt.savefig('/Users/liyurun/Desktop/figure/' + str(data[num][0][0]) + '.jpg')


##========================================
###############   run   ##################
##========================================

def main():
    for i in range(len(data)):
        if len(data[i][0][2]) > 1:
            GA_run(i, 0, 0.2, 0.03)


main()
