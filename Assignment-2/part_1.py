import numpy as np

# U(t+1)(i) = max_A[R(i,A) + gamma * SIGMA [P(j| i,A) * U(t)(j)]]
# P(t+1)(i) = argmax_A[R(i,A) + gamma * SIGMA [P(j| i,A) * U(t+1)(j)]]

Penalty = -5
Gamma = 0.99
Delta = 1e-3
Final_reward = 10

max_MD_health = 4
max_arrows_cnt = 3
max_hero_stamina = 2 


Ucurr = np.zeros((5,4,3))
Uprev = np.zeros((5,4,3))
Reward = np.zeros((5,4,3))
Policy = np.array([[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']],[['a','a','a'],['a','a','a'],['a','a','a'],['a','a','a']]],dtype = 'object')
print (Policy)

for i in range(5):
    for j in range(4):
        for k in range(3):
            if(i != 0):
                Reward[i][j][k] = Penalty
            else:
                Reward[i][j][k] = Penalty + Final_reward

def reward_shoot(MD_health , arrows_cnt , hero_stamina):
    
    prob_hit = 0.5
    return (prob_hit*Reward[MD_health-1][arrows_cnt-1][hero_stamina-1] + (1-prob_hit)*Reward[MD_health][arrows_cnt-1][hero_stamina-1])

def reward_dodge(MD_health , arrows_cnt , hero_stamina):
    
    if (hero_stamina == 2 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 0.8
        prob_pick_arrow = 0.8
        return (prob_stamina_reduce_by_50*prob_pick_arrow*Reward[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*prob_pick_arrow*Reward[MD_health][arrows_cnt+1][hero_stamina-2] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-2])          

    elif (hero_stamina == 2 and arrows_cnt == 3):
        prob_pick_arrow = 0
        prob_stamina_reduce_by_50 = 0.8
        return (prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-2])          

    elif (hero_stamina == 1 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0.8
        return (prob_stamina_reduce_by_50*prob_pick_arrow*Reward[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1])          

    elif (hero_stamina ==1 and arrows_cnt == 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0  
        return (prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Reward[MD_health][arrows_cnt][hero_stamina-1])

def reward_recharge(MD_health , arrows_cnt , hero_stamina):

    if (hero_stamina == 2):
        prob_recharge = 0
        return ((1-prob_recharge)*Reward[MD_health][arrows_cnt][hero_stamina])
    
    elif (hero_stamina < 1):
        prob_recharge = 0.8
        return (prob_recharge*Reward[MD_health][arrows_cnt][hero_stamina+1] + (1-prob_recharge)*Reward[MD_health][arrows_cnt][hero_stamina])


def utility_shoot(Uprev , MD_health , arrows_cnt , hero_stamina):

    prob_hit = 0.5

    if (arrows_cnt > 0 and hero_stamina > 0):
        return (reward_shoot(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_hit*Uprev[MD_health-1][arrows_cnt-1][hero_stamina-1] + (1-prob_hit)*Uprev[MD_health][arrows_cnt-1][hero_stamina-1]))

def utility_dodge(Uprev, MD_health , arrows_cnt , hero_stamina):
    
    if (hero_stamina == 2 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 0.8
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Uprev[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*prob_pick_arrow*Uprev[MD_health][arrows_cnt+1][hero_stamina-2] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 2 and arrows_cnt == 3):
        prob_pick_arrow = 0
        prob_stamina_reduce_by_50 = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 1 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Uprev[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1]))           

    elif (hero_stamina ==1 and arrows_cnt == 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0  
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Uprev[MD_health][arrows_cnt][hero_stamina-1]))


def utility_recharge(Uprev, MD_health , arrows_cnt , hero_stamina):

    if (hero_stamina == 2):
        prob_recharge = 0
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*((1-prob_recharge)*Uprev[MD_health][arrows_cnt][hero_stamina]))
    
    elif (hero_stamina < 1):
        prob_recharge = 0.8
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_recharge*Uprev[MD_health][arrows_cnt][hero_stamina+1] + (1-prob_recharge)*Uprev[MD_health][arrows_cnt][hero_stamina]))

def policy_shoot(Ucurr, MD_health , arrows_cnt , hero_stamina):

    prob_hit = 0.5

    if (arrows_cnt > 0 and hero_stamina > 0 and MD_health > 0):
        return (reward_shoot(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_hit*Ucurr[MD_health-1][arrows_cnt-1][hero_stamina-1] + (1-prob_hit)*Ucurr[MD_health][arrows_cnt-1][hero_stamina-1]))

def policy_dodge(Ucurr , MD_health , arrows_cnt , hero_stamina):
    
    if (hero_stamina == 2 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 0.8
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Ucurr[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*prob_pick_arrow*Ucurr[MD_health][arrows_cnt+1][hero_stamina-2] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 2 and arrows_cnt == 3):
        prob_pick_arrow = 0
        prob_stamina_reduce_by_50 = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1] + (1-prob_stamina_reduce_by_50)*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-2]))          

    elif (hero_stamina == 1 and arrows_cnt < 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0.8
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*prob_pick_arrow*Ucurr[MD_health][arrows_cnt+1][hero_stamina-1] + prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1]))           

    elif (hero_stamina ==1 and arrows_cnt == 3):
        prob_stamina_reduce_by_50 = 1
        prob_pick_arrow = 0  
        return (reward_dodge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_stamina_reduce_by_50*(1-prob_pick_arrow)*Ucurr[MD_health][arrows_cnt][hero_stamina-1]))          


def policy_recharge(Ucurr , MD_health , arrows_cnt , hero_stamina):

    if (hero_stamina == 2):
        prob_recharge = 0
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*(1-prob_recharge)*Ucurr[MD_health][arrows_cnt][hero_stamina]))
    
    elif (hero_stamina < 1):
        prob_recharge = 0.8
        return (reward_recharge(MD_health , arrows_cnt , hero_stamina) + Gamma*(prob_recharge*Ucurr[MD_health][arrows_cnt][hero_stamina+1] + (1-prob_recharge)*Ucurr[MD_health][arrows_cnt][hero_stamina]))

def can_shoot(MD_health , arrows_cnt , hero_stamina):
    if (arrows_cnt > 0 and hero_stamina > 0):
        return 1
    else:
        return 0

def can_dodge(MD_health , arrows_cnt , hero_stamina):
    if (hero_stamina > 0):
        return 1
    else:
        return 0

def value_iteration(Uprev , Ucurr)
    for i in range (max_MD_health):
        for j in range(max_arrows_cnt):
            for k in range(max_hero_stamina):
                if (max_MD_health > 0):
                    if(can_dodge and can_shoot):
                        Ucurr[i][j][k] = max(utility_shoot(Uprev,i,j,k) , utility_dodge(Uprev,i,j,k) , utility_recharge(Uprev,i,j,k))
                        policy = [policy_shoot(Ucurr,i,j,k),policy_dodge(Ucurr,i,j,k),policy_recharge(Ucurr,i,j,k)]
                        index = policy.index(max(policy))
                        if(index == 0):
                            Policy[i][j][k] = 'SHOOT'
                        elif(index == 1):
                            Policy[i][j][k] = 'DODGE'
                        elif(index == 2):
                            Policy[i][j][k] = 'RECHARGE'

                    elif(can_shoot):
                        Ucurr[i][j][k] = max(utility_shoot(Uprev,i,j,k) , utility_recharge(Uprev,i,j,k))
                        policy = [policy_shoot(Ucurr,i,j,k),policy_recharge(Ucurr,i,j,k)]
                        index = policy.index(max(policy))
                        if(index == 0):
                            Policy[i][j][k] = 'SHOOT'
                        if(index == 1):
                            Policy[i][j][k] = 'RECHARGE'

                    elif(can_dodge):
                        Ucurr[i][j][k] = max(utility_dodge(Uprev,i,j,k) , utility_recharge(Uprev,i,j,k))
                        policy = [policy_dodge(Ucurr,i,j,k),policy_recharge(Ucurr,i,j,k)]
                        index = policy.index(max(policy))
                        if(index == 0):
                            Policy[i][j][k] = 'DODGE'
                        elif(index == 1):
                            Policy[i][j][k] = 'RECHARGE'
                    else:
                        Ucurr[i][j][k] = 0
                        Policy[i][j][k] = 'NONE'    
                else:
                    Ucurr[i][j][k] = 0
                    Policy[i][j][k] = 'NONE'

    #  print krwa dio yaha pr

    check(Uprev , Ucurr)


def check(Uprev, Ucurr):

    cnt = 0
    for i in range(max_MD_health):
        for j in range(max_arrows_cnt):
            for k in range(max_hero_stamina):
                if ( abs(Ucurr[i][j][k] - Uprev[i][j][k]) < Delta):
                    cnt += 1;

    if (cnt == 60):
        value_iteration(Uprev , Ucurr)

    else:
        exit(1)    


value_iteration(Uprev,Ucurr)






