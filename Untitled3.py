#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


def rndom(n): 
    return np.random.uniform(-1, 1, size = n)

tot_runs = 1000
iterations_total = 0
mismatch_total = 0

for run in range(RUNS): 
    x = rnd(2)
    y = rnd(2)
    m = (y[1] - x[1]) / (y[0] - x[0])
    b = y[1] - m * y[0]  
    w_f = np.array([b, m, -1])
    N = 100
    X = np.transpose(np.array([np.ones(N), rndom(N), rndom(N)]))           
    y_f = np.sign(np.dot(X, w_f))                                      
    w_h = np.zeros(3)                   
    t = 0                               
    
    while True:
        
        y_h = np.sign(np.dot(X, w_h))   
        comp = (y_h != y_f)                 
        wrong = np.where(comp)[0]       
        if wrong.size == 0:
            break
        
        rnd_choice = np.random.choice(wrong)    

        
        w_h = w_h +  y_f[rnd_choice] * np.transpose(X[rnd_choice])
        t += 1

    iterations_total += t
    N_outside = 1000
    test_x0 = np.random.uniform(-1,1,N_outside)
    test_x1 = np.random.uniform(-1,1,N_outside)

    X = np.array([np.ones(N_outside), test_x0, test_x1]).T

    y_target = np.sign(X.dot(w_f))
    y_hypothesis = np.sign(X.dot(w_h))
    
    ratio_mismatch = ((y_target != y_hypothesis).sum()) / N_outside
    ratio_mismatch_total += ratio_mismatch

#------------------------------------------------
    
print("Size of training data: N = ", N, "points")
    
iterations_avg = iterations_total / RUNS
print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)

ratio_mismatch_avg = ratio_mismatch_total / RUNS
print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
print("P(f(x)!=h(x)) = ", ratio_mismatch_avg)


# In[3]:





# In[1]:


n,p = 10,0.5


# In[12]:


s = sum(np.random.binomial(n,p, 20)==0)/20


# In[13]:


print(s)


# In[ ]:




