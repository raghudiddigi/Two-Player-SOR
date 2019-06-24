
# coding: utf-8

# In[ ]:


def value_iteration(states,actions,discount,P,R):
    V = np.zeros(states) #Initial value
    iterations = 1000    
        
    for iters in range(iterations):
        Q = np.zeros((actions,actions,states))
        for a1 in range(actions):
            for a2 in range(actions):
                Q[a1,a2] = R[a1,a2] + discount * P[a1,a2].dot(V)

        v_prev = deepcopy(V)
        #print(v_prev)
        for s in range(states):
            #print(Q[:,:,s])
            rps = nash.Game(Q[:,:,s])
            #print(rps)
            eqs = rps.lemke_howson(0)
            #print(list(eqs))
            V[s] = rps[list(eqs)][0]
            #print(rps[list(eqs)])

        #print(v_prev)
        #print(V)
        #print(v_prev)
        #print(np.linalg.norm(V-v_prev))

        if np.linalg.norm(V-v_prev) < 0.00001:
            break
    return V

    


# In[ ]:


def minimax_Q(states,actions,discount,P,R,V,max_iterations):
    
    np.random.seed(100)
    
    Q = np.random.rand(states,actions,actions)
    s = np.random.randint(0, states)
    standard_norm_diff = np.zeros((max_iterations,1))

    for n in range(max_iterations):

        if (n % 100) == 0:
            s = np.random.randint(0, states)

        a1 = random.randint(0,actions-1)
        a2 = random.randint(0,actions-1)


        p_s_new = np.random.random()
        p = 0
        s_new = -1
        while (p < p_s_new) and (s_new < (states - 1)):
            s_new = s_new + 1
            #print(a1,a2,s,s_new)
            p = p + P[a1][a2][s][s_new]

        r = R[a1][a2][s]

        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]

        delta = r + discount * next_state_value - Q[s, a1,a2]
        dQ = (1 / math.sqrt(n + 2)) * delta
                #dQ = 0.001*delta
        Q[s, a1,a2] = Q[s, a1,a2] + dQ

        minimax_Q = np.zeros(states)
        for i in range(states):
            rps = nash.Game(Q[i,:,:])
            #print(rps)
            eqs = rps.lemke_howson(0)
            minimax_Q[i] = (rps[list(eqs)][0])

        standard_norm_diff[n] = np.linalg.norm(V - minimax_Q)

    return standard_norm_diff,np.linalg.norm(V - minimax_Q)


# In[ ]:


def minimax_SOR_Q(states,actions,discount,w,P,R,V,max_iterations):
    
    np.random.seed(100)
    
    #SOR Q-learning
    Q = np.random.rand(states,actions,actions)
    s = np.random.randint(0, states)
    sor_norm_diff = np.zeros((max_iterations,1))
    

    for n in range(max_iterations):

        if (n % 100) == 0:
            s = np.random.randint(0, states)

        a1 = random.randint(0,actions-1)
        a2 = random.randint(0,actions-1)


        p_s_new = np.random.random()
        p = 0
        s_new = -1
        while (p < p_s_new) and (s_new < (states - 1)):
            s_new = s_new + 1
            #print(a1,a2,s,s_new)
            p = p + P[a1][a2][s][s_new]

        r = R[a1][a2][s]

        #print(Q[s_new,:,:])


        rps = nash.Game(Q[s,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        current_state_value = rps[list(eqs)][0]


        rps = nash.Game(Q[s_new,:,:])
        #print(rps)
        eqs = rps.lemke_howson(0)
        next_state_value = rps[list(eqs)][0]

        delta = r + discount * next_state_value - Q[s, a1,a2]
        delta = w *(r + discount* next_state_value) + (1-w)*current_state_value - Q[s, a1,a2]
        dQ = (1 / math.sqrt(n + 2)) * delta
                #dQ = 0.001*delta
        Q[s, a1,a2] = Q[s, a1,a2] + dQ

        sor_minimax_Q = np.zeros(states)
        
        for i in range(states):
            rps = nash.Game(Q[i,:,:])
            #print(rps)
            eqs = rps.lemke_howson(0)
            sor_minimax_Q[i] =  rps[list(eqs)][0]
        
        sor_norm_diff[n] = np.linalg.norm(V - sor_minimax_Q) 
        

    return sor_norm_diff, np.linalg.norm(V - sor_minimax_Q)  


# In[ ]:


def code_run(count,standard_diffs,sor_diffs):

    np.random.seed((count+1)*100)
    random.seed((count+1)*110)

    P = np.zeros((actions,actions,states,states))
    R = np.random.random((actions,actions,states))
    
    const = np.random.randint(2,6) #For esnuring positive probability and w>1. 

    for a1 in range(actions):
        for a2 in range(actions):
            for s in range(states):
                P[a1,a2,s] = np.random.random(states)
                P[a1][a2][s][s] = const
                P[a1,a2,s] = P[a1,a2,s] / P[a1,a2,s].sum()
                
    
    w = 100
        
    for a1 in range(actions):
        for a2 in range(actions):
            for s in range(states):
                temp = 1/(1 - (discount*P[a1][a2][s][s]))
                if w > temp:
                    w = temp
    #print(w)
        
    
    
    

    V = value_iteration(states,actions,discount,P,R)
    standard_diffs[count], standard_last_diff = minimax_Q(states,actions,discount,P,R,V,max_iterations)
    
#     sor_diffs_a[count], sor_last_diff_a = minimax_SOR_Q(states,actions,discount,1.1,P,R,V,max_iterations)
#     sor_diffs_b[count], sor_last_diff_b = minimax_SOR_Q(states,actions,discount,1.2,P,R,V,max_iterations)
#     sor_diffs_c[count], sor_last_diff_c = minimax_SOR_Q(states,actions,discount,1.3,P,R,V,max_iterations)

    sor_diffs[count], sor_last_diff = minimax_SOR_Q(states,actions,discount,w,P,R,V,max_iterations)
    print(standard_last_diff,sor_last_diff)
    
    


# In[ ]:


import numpy as np
import nashpy as nash
from copy import deepcopy
import math
import random
import time
import multiprocessing 

manager = multiprocessing.Manager()
standard_diffs = manager.dict()
sor_diffs = manager.dict()

starttime = time.time()

total_mdps = 50

states = 10
actions = 5
discount = 0.6
max_iterations = 100000

# standard_diffs = np.zeros((total_mdps,max_iterations,1))
# sor_diffs = np.zeros((total_mdps,max_iterations,1))



# sor_diffs_a = np.zeros((total_mdps,max_iterations))
# sor_diffs_b = np.zeros((total_mdps,max_iterations))
# sor_diffs_c = np.zeros((total_mdps,max_iterations))
# sor_diffs_d = np.zeros((total_mdps,max_iterations))

processes = []

for count in range(total_mdps):
    
    p = multiprocessing.Process(target=code_run, args=(count,standard_diffs,sor_diffs))
    processes.append(p)
    p.start()

for process in processes:
        process.join()
    
    
np.savetxt('minmax-normal',np.average(standard_diffs.values(),axis = 0))
np.savetxt('minmax-sor',np.average(sor_diffs.values(),axis = 0))

print('That took {} seconds'.format(time.time() - starttime))
            

