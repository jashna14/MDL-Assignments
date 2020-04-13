import cvxpy as cp
import numpy as np
import json
import os

key = {}
alpha = []
alpha_new = []
reward = -20
r = []
solutionx = []
policy = []

NOOP = 1
SHOOT = 2
DODGE = 3
RECHARGE = 4

action = {
	'1' : 'NOOP',
	'2' : 'SHOOT',
	'3' : 'DODGE',
	'4' : 'RECHARGE'
 } 

def construct_key(cnt):
	for i in range(5):
		for j in range(4):
			for k in range(3):
				key[str(i) + str(j) + str(k)] = [cnt,[],[],[],[]]
				cnt += 1




def shoot(md_health, arrows_cnt, stamina):
	if(arrows_cnt > 0 and md_health > 0 and stamina > 0):
		hit_prob = 0.5

		key[str(md_health) + str(arrows_cnt) + str(stamina)][SHOOT].append([str(md_health-1) + str(arrows_cnt - 1) + str(stamina - 1), hit_prob])
		key[str(md_health) + str(arrows_cnt) + str(stamina)][SHOOT].append([str(md_health) + str(arrows_cnt - 1) + str(stamina - 1), 1 - hit_prob])


def recharge(md_health, arrows_cnt, stamina):
	if(stamina < 2 and md_health > 0):
		prob = 0.8

		key[str(md_health) + str(arrows_cnt) + str(stamina)][RECHARGE].append([str(md_health) + str(arrows_cnt) + str(stamina + 1), prob])
		# key[str(md_health) + str(arrows_cnt) + str(stamina)][RECHARGE].append([str(md_health) + str(arrows_cnt) + str(stamina), 1-prob])

def dodge(md_health, arrows_cnt, stamina):
	if(md_health > 0):
		if(stamina == 2 and arrows_cnt < 3):
			prob_stamina_reduction_by_50 = 0.8
			prob_pick_arrow = 0.8
			
			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt+1) + str(stamina-1),0.64])
			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt) + str(stamina-1),0.16])
			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt+1) + str(stamina -2),0.16])
			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt) + str(stamina - 2),0.04])

		if(stamina == 2 and arrows_cnt == 3):
			prob_stamina_reduction_by_50 = 0.8
			prob_pick_arrow = 0

			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt) + str(stamina-1),0.8])
			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt) + str(stamina - 2),0.2])


		if(stamina == 1 and arrows_cnt < 3):
			prob_stamina_reduction_by_50 = 1
			prob_pick_arrow = 0.8

			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt+1) + str(stamina-1),0.8])
			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt) + str(stamina-1),0.2])

		if(stamina == 1 and arrows_cnt == 3):
			prob_stamina_reduction_by_50 = 1
			prob_pick_arrow = 0

			key[str(md_health) + str(arrows_cnt) + str(stamina)][DODGE].append([str(md_health) + str(arrows_cnt) + str(stamina-1),1])

def noop(md_health, arrows_cnt, stamina):
	if(md_health == 0):
		noop_prob = 1

		key[str(md_health) + str(arrows_cnt) + str(stamina)][NOOP].append([str(md_health) + str(arrows_cnt) + str(stamina),noop_prob])

def fill_key():
	for i in range(5):
		for j in range(4):
			for k in range(3):
				noop(i,j,k)
				shoot(i,j,k)
				recharge(i,j,k)
				dodge(i,j,k)

def count_A_matrix_hori_dimension(cnt):
	for i in range(5):
		for j in range(4):
			for k in range(3):
				for l in range(1,5):
					if len(key[str(i) + str(j) + str(k)][l]) > 0:
						cnt += 1

	return cnt

def fill_A_matrix():
	cnt = 0
	for i in range(5):
		for j in range(4):
			for k in range(3):
				for l in range(1,5):
					if len(key[str(i) + str(j) + str(k)][l]) > 0:
						for m in (key[str(i) + str(j) + str(k)][l]):
							if(i != 0):
								A[key[str(i) + str(j) + str(k)][0]][cnt] += m[1]
								A[key[m[0]][0]][cnt] -= m[1]	
							if(i == 0):
								A[key[str(i) + str(j) + str(k)][0]][cnt] += m[1]
								
						cnt += 1

def construct_alpha():
	for i in range(60-1):
		alpha.append([0])
	alpha.append([1])


def construct_r():
	for i in range(5):
		for j in range(4):
			for k in range(3):
				for l in range(1,5):
					if len(key[str(i) + str(j) + str(k)][l]) > 0:
						if(l == 1):
							r.append(0)
						else:
							r.append(reward)

def get_new_alpha():
	for i in range(60-1):
		alpha_new.append(0.0)
	alpha_new.append(1.0)	



def get_output(r,alpha):
	r = np.array(r)
	alpha = np.array(alpha)
	x = cp.Variable(shape=(100,1), name="x")
	constraints = [cp.matmul(A, x) == alpha, x>=0]
	objective = cp.Maximize(cp.matmul(r,x))
	problem = cp.Problem(objective, constraints)
	solution = problem.solve()
	for i in range(len(x.value)):
	    solutionx.append(x.value[i][0])

	return solution

def get_policy():
	count = 0
	for i in range(5):
		for j in range(4):
			for k in range(3):
				x = []
				for l in range(1,5):
					if len(key[str(i) + str(j) + str(k)][l]) > 0:
						x.append(l)


				arr = []		
				for l in range(0,len(x)):
					arr.append(solutionx[count + l])
				count += len(x)	


				maxactionpos = arr.index(max(arr))
				policy.append([[i,j,k],action[str(x[maxactionpos])]])


cnt = 0
construct_key(cnt)
fill_key()
cnt = 0
A_hori_dimension = count_A_matrix_hori_dimension(cnt) 
A_ver_dimension = 5*4*3
A = np.zeros((A_ver_dimension,A_hori_dimension))
fill_A_matrix()
construct_alpha()
construct_r()
objective = get_output(r,alpha)
get_policy()
get_new_alpha()

output = {}

output["a"] = A.tolist()
output["r"] = r
output["alpha"] = alpha_new
output["x"] = solutionx
output["policy"] = policy
output["objective"] = objective

if not os.path.exists('outputs'):
	os.mkdir('outputs')
    
with open('outputs/output.json','w') as f:
	json.dump(output,f)

