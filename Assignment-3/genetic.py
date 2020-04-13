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
#
pop_size=10
radiation=2.5

population=np.zeros((pop_size,11))
new_population=np.zeros((pop_size,11))
fitness=np.zeros((pop_size))
new_fitness=np.zeros((pop_size))
probability=np.zeros((pop_size))
indices=np.zeros((pop_size))
for i in range(pop_size):
    population[i]=list(data)
    indices[i]=i

# data[0]-=20
# data[1]-=0.5
# data[2]-=0.19
# data[3]+=0.004
# data[4]-=0.000008
# data[5]-=0.00000009
# data[6]+=0.000000001
# data[7]+=0.0000000001
# data[8]-=0.000000000001
# data[9]-=0.00000000000001
# data[10]+=0.000000000000001
# print(data)

# print(data)
# print(get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",data))

for i in range(pop_size-1):
    mutation=random.uniform(-radiation,radiation)
    victim=random.randint(0,10)
    mutation/=(10**((victim*2)-4))
    population[i][victim]+=mutation
    if(population[i][victim]>10):
        population[i][victim]=10
    elif(population[i][victim]<-10):
        population[i][victim]=-10

# prob=[0.5,0,0,0.5,0,0,0,0,0,0]
# index=(np.random.choice(indices,p=prob))
# print(index)

# print(indices*2)
# print(get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",data))
   
def get_fitness(training,validation):
    # return 1/((((validation)+(training))**2)*((validation-training)**5))
    # return 1/(((validation - training)**2+0.00000000000001)*((validation+training)**3))
    # return (1/((training+3*validation))+1/((1.5*training)+(1.5*validation)))
    # if(((1.3 - training/validation) + 1800000/(training+validation)) < 0):
    #     return 0
    # else:
    return 1/((validation+2*training))

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

    # training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",father.tolist())
    # fit_father=get_fitness(training,validation)
    # training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",mother.tolist())
    # fit_mother=get_fitness(training,validation)
    
    # if(fit_father>fit_mother):
    #     return father
    # else:
    #     return mother


def mutate():
    global radiation
    global population
    for i in range(pop_size):
        if(probability[i]<=1):
            mutation=random.uniform(-radiation,radiation)
            victim=random.randint(0,10)
            mutation/=(10**((victim*2)-4))
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
        # max1=0
        # max2=0
        # for k in range(pop_size):
        #     if(probability[k]>probability[max1]):
        #         max2=max1
        #         max1=k
        #     elif(probability[k]>probability[max2]):
        #         max2=k
        
        # father=population[int(max1)]
        # mother=population[int(max2)]
        child=reproduce(father,mother)
        new_population[i]=child


    # print(population)
    # print(new_population)
    for i in range(pop_size):
        training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",new_population[index].tolist())
        new_fitness[i]=get_fitness(training,validation)


    # population=new_population

    index1 = np.argpartition(fitness, -5)[-5:] 
    index2 = np.argpartition(new_fitness, -5)[-5:]

    p=np.zeros((pop_size,11))

    for i in range(0,5):
        p[i] = population[index1[i]]
    for i in range(0,5):
        p[i+5] = new_population[index2[i]]

    population = p         
    # get_probabilites()
    mutate()
    get_probabilites()


print(probability,population,fitness)

for i in range(pop_size):
    training,validation=get_errors("mBAkj2CeFNwihROmN2lzWnH6EJ9uBAXQGBxUD4hnRDKzm1BWkm",population[i].tolist())
    print(training,validation)

# # test()
# # print(probability)
# # print(population)
# # mutate()
# # print("fuck")
# #   print(population[2][3])



# [ 4.11620141e-01, 8.28840025e-01, -6.32813312e+00, 3.98981302e-02, 3.84598946e-02, 7.86338239e-05, -6.01731557e-05, -1.25158557e-07, 3.48409638e-08, 4.16173232e-11, -6.73246633e-12]


# BEST AS OF 26 /03/2020[before] [-1.00000000e+01, 1.00000000e+01, -5.81874693e+00, 5.54036262e-02, 3.79238979e-02, 8.12934758e-05, -6.01120733e-05, -1.25744629e-07, 3.48555581e-08, 3.90697687e-11, -6.73041924e-12]
