from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
r = random.Random()
r.seed("AI")


import math


# region SearchAlgorithms
class Stack:

    def __init__(self):
        self.stack = []

    def push(self, value):
        if value not in self.stack:
            self.stack.append(value)
            return True
        else:
            return False

    def exists(self, value):
        if value not in self.stack:
            return True
        else:
            return False

    def pop(self):
        if len(self.stack) <= 0:
            return ("The Stack == empty")
        else:
            return self.stack.pop()

    def top(self):
        return self.stack[0]


class Node:
    id = None
    comefrom = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    edgeCost = None
    gOfN = None  # total edge cost
    hOfN = None  # heuristic value
    heuristicFn = None

    def __init__(self, value):
        self.value = value



class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    totalCost = 0
    listNodes = []
    k = 0
    row = None
    colum = None
    queue=[]
    end = 0
    start = 0

    def __init__(self, mazeStr,edgeCost = None):
        ''' mazeStr contains the full board
         The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''

        self.row=mazeStr.split(' ')
        self.colum=int((len(self.row[0])+1)/2)
        i = 0
        nodes=[]
        for x in mazeStr:
                if x == ' ':
                    self.k=0
                    self.listNodes.append(nodes)
                    nodes=[]
                    i+=1
                elif x == '.':
                    node = Node('.')
                    node.id = (self.colum* i) + (self.k)
                    node.edgeCost = edgeCost[(self.colum*i)+self.k]
                    self.k+= 1
                    nodes.append(node)
                elif x == 'S':
                    node = Node('S')
                    node.id = (self.colum* i) + (self.k)
                    node.edgeCost = edgeCost[(self.colum*i)+self.k]
                    self.k = self.k +1
                    nodes.append(node)
                    self.start = (self.colum* i) + (self.k)
                elif x == 'E':
                    node = Node('E')
                    node.id = (self.colum* i) + (self.k)
                    node.edgeCost = edgeCost[(self.colum * i) + self.k]
                    self.k += 1
                    nodes.append(node)
                    self.end = (self.colum* i) + (self.k)
                elif x == '#':
                    node = Node('#')
                    node.id = (self.colum* i) + (self.k)
                    node.edgeCost = edgeCost[(self.colum* i) + self.k]
                    self.k += 1
                    nodes.append(node)

        self.listNodes.append(nodes)
        x1 = int(self.end/self.colum)
        y1 = (self.end % self.colum)-1

        for i in range(len(self.row)):
            for j in range(self.colum):
              self.listNodes[i][j].hOfN = abs(i-x1)+abs(j-y1)
              if i!=0:
                 self.listNodes[i][j].up=self.listNodes[i-1][j]
              if i!=len(self.row)-1:
                 self.listNodes[i][j].down=self.listNodes[i+1][j]
              if j!=0:
                 self.listNodes[i][j].left=self.listNodes[i][j-1]
              if j != self.colum-1:
                 self.listNodes[i][j].right=self.listNodes[i][j+1]
    pass

    def AstarManhattanHeuristic(self):
        self.queue.append(self.listNodes[int(self.start/self.colum)][(self.start % self.colum)-1])
        node=Node('')
        while len(self.queue)!=0:
               node=self.queue.pop(0)
               self.fullPath.append(node.id)
               if node.value =='E':
                   break
               if node.up != None and node.comefrom != 'up' and node.up.value!='#':
                   node.up.gOfN=node.edgeCost+node.up.edgeCost
                   node.up.heuristicFn=node.up.hOfN+node.up.gOfN
                   node.up.comefrom='down'
                   node.up.previousNode=node
                   self.queue.append(node.up)
               if node.down != None and node.comefrom != 'down' and node.down.value!='#':
                   node.down.gOfN = node.edgeCost + node.down.edgeCost
                   node.down.heuristicFn = node.down.hOfN + node.down.gOfN
                   node.down.comefrom='up'
                   node.down.previousNode = node
                   self.queue.append(node.down)
               if node.left != None and node.comefrom != 'left' and node.left.value!='#':
                   node.left.gOfN = node.edgeCost + node.left.edgeCost
                   node.left.heuristicFn = node.left.hOfN + node.left.gOfN
                   node.left.comefrom='right'
                   node.left.previousNode = node
                   self.queue.append(node.left)
               if node.right != None and node.comefrom != 'right' and node.right.value!='#':
                   node.right.gOfN = node.edgeCost + node.right.edgeCost
                   node.right.heuristicFn = node.right.hOfN + node.right.gOfN
                   node.right.comefrom='left'
                   node.right.previousNode = node
                   self.queue.append(node.right)
               self.queue.sort(key=lambda Node:Node.heuristicFn)

        self.path.append(node.id)
        while node.previousNode!=None :
            self.path.append(node.previousNode.id)
            self.totalCost+=node.edgeCost
            node = node.previousNode
        self.path.reverse()
        return self.fullPath, self.path, self.totalCost






# endregion

# region KNN
class KNN_Algorithm:
    list=[]
    list2=[]
    counter1=0
    counter2=0
    correct=0
    def __init__(self, K):
        self.K = K

    def euclidean_distance(self, p1, p2):

        sum = 0
        for i in range(len(p1)):
            sum += (p1[i] - p2[i]) ** 2
        return sqrt(sum)


    def KNN(self, X_train, X_test, Y_train, Y_test):
      for i in range(len(X_test)):
          for j in range(len (X_train)):
            d=self.euclidean_distance(X_test[i],X_train[j])
            self.list.append((Y_train[j],d))
          self.list.sort(key=lambda x:x[1])
          for i in range(self.K):
              if self.list[i][0]==1:
                  self.counter1+=1
              else:
                  self.counter2+=1
          if self.counter1>self.counter2:
              self.list2.append(1)
          else:
              self.list2.append(0)
          self.list=[]
          self.counter1=0
          self.counter2=0
      for i in range(len(self.list2)):
          if self.list2[i]==Y_test[i]:
              self.correct+=1
      accurecy=self.correct/len(Y_test)*100
      return accurecy

# endregion


# region GeneticAlgorithm
class GeneticAlgorithm:
    Cities = [1, 2, 3, 4, 5, 6]
    DNA_SIZE = len(Cities)
    POP_SIZE = 20
    GENERATIONS = 5000

    """
    - Chooses a random element from items, where items is a list of tuples in
       the form (item, weight).
    - weight determines the probability of choosing its respective item. 
     """

    def weighted_choice(self, items):
        weight_total = sum((item[1] for item in items))
        n = r.uniform(0, weight_total)
        for item, weight in items:
            if n < weight:
                return item
            n = n - weight
        return item

    """ 
      Return a random character between ASCII 32 and 126 (i.e. spaces, symbols, 
       letters, and digits). All characters returned will be nicely printable. 
    """

    def random_char(self):
        return chr(int(r.randrange(32, 126, 1)))

    """ 
       Return a list of POP_SIZE individuals, each randomly generated via iterating 
       DNA_SIZE times to generate a string of random characters with random_char(). 
    """

    def random_population(self):
        pop = []
        for i in range(1, 21):
            x = r.sample(self.Cities, len(self.Cities))
            if x not in pop:
                pop.append(x)
        return pop

    """ 
      For each gene in the DNA, this function calculates the difference between 
      it and the character in the same position in the OPTIMAL string. These values 
      are summed and then returned. 
    """

    def cost(self, city1, city2):
        if (city1 == 1 and city2 == 2) or (city1 == 2 and city2 == 1):
            return 10
        elif (city1 == 1 and city2 == 3) or (city1 == 3 and city2 == 1):
            return 20
        elif (city1 == 1 and city2 == 4) or (city1 == 4 and city2 == 1):
            return 23
        elif (city1 == 1 and city2 == 5) or (city1 == 5 and city2 == 1):
            return 53
        elif (city1 == 1 and city2 == 6) or (city1 == 6 and city2 == 1):
            return 12
        elif (city1 == 2 and city2 == 3) or (city1 == 3 and city2 == 2):
            return 4
        elif (city1 == 2 and city2 == 4) or (city1 == 4 and city2 == 2):
            return 15
        elif (city1 == 2 and city2 == 5) or (city1 == 5 and city2 == 2):
            return 32
        elif (city1 == 2 and city2 == 6) or (city1 == 6 and city2 == 2):
            return 17
        elif (city1 == 3 and city2 == 4) or (city1 == 4 and city2 == 3):
            return 11
        elif (city1 == 3 and city2 == 5) or (city1 == 5 and city2 == 3):
            return 18
        elif (city1 == 3 and city2 == 6) or (city1 == 6 and city2 == 3):
            return 21
        elif (city1 == 4 and city2 == 5) or (city1 == 5 and city2 == 4):
            return 9
        elif (city1 == 4 and city2 == 6) or (city1 == 6 and city2 == 4):
            return 5
        else:
            return 15

    # complete fitness function
    def fitness(self, dna):
     fitness=0
     for i in range(self.DNA_SIZE):
         if i==self.DNA_SIZE-1:
            fitness+=self.cost(dna[i],dna[0])
         else:
             fitness+=self.cost(dna[i],dna[i+1])
     return fitness
    """ 
       For each gene in the DNA, there is a 1/mutation_chance chance that it will be 
       switched out with a random character. This ensures diversity in the 
       population, and ensures that is difficult to get stuck in local minima. 
       """

    def mutate(self, dna, random1, random2):
       if random1 <= 0.01:
         for i in range(self.DNA_SIZE):

             tmp=dna[int(random2*self.DNA_SIZE)]
             dna[int(random2*self.DNA_SIZE)]=dna[i]
             dna[i]=tmp

       return dna


       """ 
       Slices both dna1 and dna2 into two parts at a random index within their 
       length and merges them. Both keep their initial sublist up to the crossover 
       index, but their ends are swapped. 
       """

    def crossover(self, dna1, dna2, random1, random2):
      r2 = int(random2 * self.DNA_SIZE)
      l1 = []
      l2 = []
      if random1<=0.9:
        tmp=dna1[r2:]
        tmp1=dna2[r2:]
        counter=r2
        for i in dna2:
           if i in tmp:
              l1.append(i)
              counter+=1
        counter=r2
        for i in dna1:
           if i  in tmp1:
               l2.append(i)
               counter+=1


        return dna1[:r2]+l1,dna2[:r2]+l2
      else:
          return dna1,dna2


# endregion
#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                                  [0, 15, 2, 100, 60, 35, 30, 3
                                          , 100, 2, 15, 60, 100, 30, 2
                                          , 100, 2, 2, 2, 40, 30, 2, 2
                                          , 100, 100, 3, 15, 30, 100, 2
                                          , 100, 0, 2, 100, 30])
    fullPath, path, TotalCost = searchAlgo.AstarManhattanHeuristic()
    print('**ASTAR with Manhattan Heuristic ** Full Path:' + str(fullPath) + '\nPath is: ' + str(path)
          + '\nTotal Cost: ' + str(TotalCost) + '\n\n')


# endregion

# region KNN_MAIN_FN
'''The dataset classifies tumors into two categories (malignant and benign) (i.e. malignant = 0 and benign = 1)
    contains something like 30 features.
'''


def KNN_Main():
    BC = load_breast_cancer()
    X = []

    for index, row in pd.DataFrame(BC.data, columns=BC.feature_names).iterrows():
        temp = []
        temp.append(row['mean area'])
        temp.append(row['mean compactness'])
        X.append(temp)
    y = pd.Categorical.from_codes(BC.target, BC.target_names)
    y = pd.get_dummies(y, drop_first=True)
    YTemp = []
    for index, row in y.iterrows():
        YTemp.append(row[1])
    y = YTemp;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1024)
    KNN = KNN_Algorithm(7);
    accuracy = KNN.KNN(X_train, X_test, y_train, y_test)
    print("KNN Accuracy: " + str(accuracy))


# endregion

# region Genetic_Algorithm_Main_Fn
def GeneticAlgorithm_Main():
    genetic = GeneticAlgorithm();
    population = genetic.random_population()
    for generation in range(genetic.GENERATIONS):
        # print("Generation %s... Random sample: '%s'" % (generation, population[0]))
        weighted_population = []

        for individual in population:
            fitness_val = genetic.fitness(individual)

            pair = (individual, 1.0 / fitness_val)
            weighted_population.append(pair)
        population = []

        for _ in range(int(genetic.POP_SIZE / 2)):
            ind1 = genetic.weighted_choice(weighted_population)
            ind2 = genetic.weighted_choice(weighted_population)
            ind1, ind2 = genetic.crossover(ind1, ind2, r.random(),r.random())
            population.append(genetic.mutate(ind1,r.random(),r.random()))
            population.append(genetic.mutate(ind2,r.random(),r.random()))

    fittest_string = population[0]
    minimum_fitness = genetic.fitness(population[0])
    for individual in population:
        ind_fitness = genetic.fitness(individual)
    if ind_fitness <= minimum_fitness:
        fittest_string = individual
        minimum_fitness = ind_fitness

    print(fittest_string)
    print(genetic.fitness(fittest_string))


# endregion
######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    KNN_Main()
    GeneticAlgorithm_Main()
