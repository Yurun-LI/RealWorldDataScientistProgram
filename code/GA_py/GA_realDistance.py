import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import random
import math


def load_dataset(month="2020年6月", date=20200613, grouped=True):
    root_path = "/Users/liyurun/Downloads/OneDrive_1_11-18-2020/提供データ/配達実績/" + month + ".xls"
    original = pd.read_excel(root_path)

    # 指定した日付のみを抽出
    data = original[original["完了日付"] == date]
    data = data.drop(["完了日付"], axis=1)

    # 緯度・経度情報を統合
    data_0 = data[data["緯度＿完了優先"] == 0]
    data_0["緯度"] = data_0["緯度＿住所より"]
    data_0["経度"] = data_0["経度＿住所より"]

    data_1 = data[data["緯度＿完了優先"] != 0]
    data_1["緯度"] = data_1["緯度＿完了優先"]
    data_1["経度"] = data_1["経度＿完了優先"]

    data = pd.concat([data_0, data_1], axis=0)

    # コード0を削除
    data = data[data["配達乗務員\nコード"] != 0]

    # 必要なcolumnのみ抽出
    col = ["配達乗務員\nコード", "完了時刻", "緯度", "経度", "個数", "重量"]
    data = data[col]

    # ソート
    data = data.sort_values(["配達乗務員\nコード", "完了時刻"])
    data.reset_index(drop=True, inplace=True)

    # 緯度・経度に関して重複する部分をグループ化する場合
    if grouped:
        _data = pd.DataFrame()
        for driver_id in get_id_list(data):
            data_per_driver = data[data["配達乗務員\nコード"] == driver_id]
            data_per_driver = grouped_duplicate(data_per_driver)
            _data = _data.append(data_per_driver, ignore_index=True)
        data = _data
        data.reset_index(drop=True, inplace=True)

    return data


# 緯度・経度に関して重複する部分をグループ化
def grouped_duplicate(data):
    # 緯度・経度が同じものに関してグループ化
    df_1 = data.groupby(["緯度", "経度"]).max()  # df_1:グループの中の最大値を取得
    df_2 = data.groupby(["緯度", "経度"]).min()  # df_2:グループの中の最小値を取得
    df_3 = data.groupby(["緯度", "経度"]).sum()  # df_3:グループの中の総和を取得

    df_1 = df_1[["配達乗務員\nコード", "完了時刻"]]
    df_2 = df_2[["完了時刻"]].rename(columns={"完了時刻": "1st完了時刻"})
    df_3 = df_3[["個数", "重量"]]

    # 結合
    df = pd.concat([df_1, df_2, df_3], axis=1)

    # 緯度・経度がindexに入ってるためcolumnへ
    df["緯度"] = [df.index[i][0] for i in range(len(df.index))]
    df["経度"] = [df.index[i][1] for i in range(len(df.index))]

    # 順序を入れ替える
    df = df[["配達乗務員\nコード", "1st完了時刻", "完了時刻", "緯度", "経度", "個数", "重量"]]
    df = df.sort_values("完了時刻")
    df.reset_index(drop=True, inplace=True)

    return df


# dataに含まれるドライバーのIDを取得
def get_id_list(data):
    data_grouped = data.groupby("配達乗務員\nコード")
    id_list = list(data_grouped.size().index)
    return id_list


# データから指定したIDのドライバー1人のデータを取得
def get_data_per_driver(data, driver_id):
    data_per_driver = data[data["配達乗務員\nコード"] == driver_id]
    data_per_driver.reset_index(drop=True, inplace=True)
    return data_per_driver

######################################################
######################################################
class City(object):

    ## Inicializa classe
    def __init__(self, x, y):
        self.x = x
        self.y = y

    ## Calcula distância euclidiana
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    ## Coordenadas da cidade
    #def __repr__(self):
       # return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness(object):
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):

        if self.distance == 0:
            pathDistance = 0
            ## Percorre as cidades
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None

                ## Se não tem cidade na rota, adiciona
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]

                ## acrescenta distancia do caminho
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
            #print("Fitness: ", self.fitness)
        return self.fitness


## Criar população de rotas
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


## Criar uma população
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


## Ranquear o fitness de cada rota
def rankRoutes(population):
    fitnessResults = {}

    ## loop para calcular cada fitness
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    ## retorna ordenado os fitness
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []

    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])

    ## Método da roleta
    ## Soma acumulativa dos fitness
    df['cum_sum'] = df.Fitness.cumsum()

    ## Define porcentagem de cada rota dividindo pelo total
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    ## Seleciona elite
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    ## Seleciona demais indivíduos pelo metodo da roleta
    for i in range(0, len(popRanked) - eliteSize):

        ## Chance de seleção
        pick = 100 * random.random()

        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults


## Mating Pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


## Cruzamento
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    ## Inicio e fim do corte de cruzamento
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingPool, eliteSize):
    children = []
    length = len(matingPool) - eliteSize
    pool = random.sample(matingPool, len(matingPool))

    for i in range(0, eliteSize):
        children.append(matingPool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingPool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        # SE aleatorio menor que taxa de mutacão, troca
        if (random.random() < mutationRate):
            # seleciona indice
            swapWith = int(random.random() * len(individual))

            # permuta 2 cidades no individuo
            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    ## aplica a mutação em cada individuo da população
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)

    return mutatedPop


## Define nova geração
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)

    return nextGeneration


## algoritmo genetico
def ga(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def showMap(cityList):
    print(cityList)
    prev=cityList[-1]
    for i in cityList:
        plt.plot(i.x, i.y,'ro')
        plt.plot(prev.x,prev.y, 'k-')
        if(prev.x == 0 and prev.y == 0):
            prev=i
            continue;
        else:
            plt.plot([prev.x,i.x],[prev.y, i.y],'k-')
            prev=i
    plt.show()


data = load_dataset()
df = grouped_duplicate(data)
driver_id = get_id_list(data)
driver_data = get_data_per_driver(data,driver_id[0])

lat = driver_data['緯度'].tolist()
lon = driver_data['経度'].tolist()
cityList = []
for i in range(len(lat)):
    cityList.append(City(x=lat[i], y=lon[i]))

