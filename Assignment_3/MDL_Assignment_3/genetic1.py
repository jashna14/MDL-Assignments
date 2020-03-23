import numpy as np
import random
from client_moodle import *

f=open("overfit.txt","r")
data=f.read()
data=data.rstrip()
data=data.strip('][').split(', ')
f.close()
for i in range(len(data)):
    data[i]=float(data[i])

pop_size=10
radiation=0.025

population=np.zeros((pop_size,11))
new_population=np.zeros((pop_size,11))
fitness=np.zeros((pop_size))
probability=np.zeros((pop_size))
indices=np.zeros((pop_size))
for i in range(pop_size):
    population[i]=list(data)
    indices[i]=i

# prob=[0.5,0,0,0.5,0,0,0,0,0,0]
# index=(np.random.choice(indices,p=prob))
# print(index)

# print(indices*2)
# print(get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",data))
   
def get_fitness(training,validation):
    return 1/((2*validation)+(training))

def get_probabilites():
    global fitness
    global probability
    for i in range(pop_size):
        training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",population[i].tolist())
        fitness[i]=get_fitness(training,validation)

    probability=fitness*(1/sum(fitness))

def get_mutated_probability(index):
    global fitness
    global probability
    training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",population[index].tolist())
    fitness[index]=get_fitness(training,validation)
    probability=fitness*(1/sum(fitness))
    

def reproduce(father,mother):
    start=random.randint(1,pop_size-1)
    for i in range(start,pop_size):
        temp=father[i]
        father[i]=mother[i]
        mother[i]=temp
    successor=random.randint(0,1)
    if(successor==0):
        return father
    else:
        return mother  

def mutate():
    global radiation
    global population
    for i in range(pop_size):
        if(probability[i]<=0.49):
            mutation=random.uniform(-radiation,radiation)
            victim=random.randint(0,10)
            # print(population[i][victim])
            population[i][victim]+=mutation
            
            if(population[i][victim]>10):
                population[i][victim]=10
            elif(population[i][victim]<-10):
                population[i][victim]=-10
            
            # print("population",population)
            # print()
            # print("pop[i]",population[i])
            # print()
            # print("------------------------------------------------")
            # print()
            # print(i,mutation,victim,population[i][victim])
            # print()
            # get_mutated_probability(i)

# def test():
#     global population
#     print()
#     print(population[1][0],population[2][0])
#     print()
#     population[1][0]+=10.0
#     print()
#     print(population[1][0],population[2][0])
#     print()


get_probabilites()

for j in range(100):            
    for i in range(pop_size):
        x=np.random.choice(indices,p=probability)
        father=population[int(x)]
        y=np.random.choice(indices,p=probability)
        mother=population[int(y)]
        child=reproduce(father,mother)
        new_population[i]=child


    # print(population)
    # print(new_population)
    population=new_population
    get_probabilites()
    mutate()
    get_probabilites()

print(probability,population,fitness)
for i in range(pop_size):
    training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",population[i].tolist())
    print(training,validation)

# test()
# print(probability)
# print(population)
# mutate()
# print("fuck")
#   print(population[2][3])