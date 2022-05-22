#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pulp
import pandas as pd
import openpyxl

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator


# In[2]:


# process price data, select prices with label 1

hours = list(map(str, range(24))) 
prices_columns = list(map(str, range(24))) + ['label']
prices = pd.read_csv('./TestingResults.txt', names = prices_columns)
prices = prices.loc[prices['label']==1]
prices.pop('label')


# In[3]:


# process user data, collect tasks

usertask = pd.read_excel("./COMP3217CW2Input.xlsx", sheet_name = 'User & Task ID')
usertask.pop('Maximum scheduled energy per hour')


# In[4]:


usertask.dtypes


# In[5]:


# construct lp problem for each user

for user_id in range(1,6):
    
    # lp_user1 = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)
    # ...
    # lp_user5 = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)
    exec("lp_user%s=pulp.LpProblem('My_LP_Problem', pulp.LpMinimize)"%user_id)


# In[6]:


# construct decision variables 

task_by_hour = [[[]for j in range(24)] for i in range(5)]
task_by_id = [[[]for j in range(10)] for i in range(5)]

user_id = 1
task_id = 1

for task in usertask.itertuples():
    
    # construct decision variables for each task
    # if task1 of user1 starts from 20 to 23, then u1_t1_20, u1_t1_21, u1_t1_22 and u1_t1_23 will be crated
    for hour in range(task[2],task[3]+1):
        
        # u1_t1_20 = pulp.LpVariable('u1_t1_20', lowBound=0, upBound=1, cat='Continuous')
        # ...
        # u5_t10_23 = pulp.LpVariable('u5_t10_23', lowBound=0, upBound=1, cat='Continuous')
        exec("u%s_t%s_%s=pulp.LpVariable('u%s_t%s_%s', lowBound=0, upBound=1, cat='Continuous')"%(user_id,task_id,hour,user_id,task_id,hour))
        
        # store decision variables by hours
        exec("task_by_hour[%s][%s].append(u%s_t%s_%s)"%(user_id-1,hour,user_id,task_id,hour))
        
        # store decision variables by task id
        exec("task_by_id[%s][%s].append(u%s_t%s_%s)"%(user_id-1,task_id-1,user_id,task_id,hour))
        
    task_id += 1
    if task_id == 11:
        task_id = 1
        user_id += 1
        task


# In[7]:


# construct constrain function

user_constrain = "0"
task_id = 1
user_id = 1

for task_energy in usertask['Energy Demand']:
    
    # construct constrain function for each task
    for task in task_by_id[user_id-1][task_id-1]:
        
        # add up all variables by current task id for current user
        exec("user_constrain += '+ %s'"%task)
    
    # construct constrain function (sum of variables of this task == energy demand of this task)
    exec("user_constrain += ' == %s'"%task_energy)
    
    # construct constrain function of current task for current user
    exec("lp_user%s += %s"%(user_id,user_constrain))
    
    task_id += 1
    if task_id == 11:
        task_id = 1
        user_id += 1
    user_constrain = "0"
    


# In[8]:


# create folder

def mkdir(path): 
    import os
 
    path=path.strip()
    path=path.rstrip("\\")

    if not os.path.exists(path):
        os.makedirs(path)  
        return True


# In[9]:


# plot scheduling results

def plot_schedule(energy_usage,plt_name):
    
    energy_usage.append(0)
    
    # set x-coordinate scale label
    hour_list = list(map(int, range(25)))
    
    # set dpi and size of figure
    plt.figure(dpi = 300, figsize = (15,10))

    # set y-coordinate to int
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # set y-coordinate interval to 1
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))

    # set coordinate label
    plt.xlabel('Time (H)')
    plt.ylabel('Total Power (KW)')
    
    # draw bar chart
    plt.bar(range(25), energy_usage, width = 1,align = 'edge',linewidth = 1,edgecolor = 'black' , tick_label = hour_list)

    
    # save the bar chart
    mkdir('./charts_for_abnormal_prices/')
    plt.savefig('./charts_for_abnormal_prices/energy_usage(guideline%s).jpg' % plt_name)   
    plt.close()


# In[10]:


# construct objective function and do calculation

tasklist = "0"
user_function = "0"
energy_usage = [0]*24

for price in prices.itertuples():
    
    for user_id in range(1,6):
        
        for hour in range(0,24):
            
            for task in task_by_hour[user_id-1][hour]:
                
                # add up variables in current hour for current user 
                exec("tasklist += '+ %s'"%task)
            
            # add up (price * sum of variables) of each hour for current user, to construct objective function
            exec("user_function += '+ (%s) * %s'"%(tasklist,price[hour+1]))
            
            # reset the sum of variables of current hour
            tasklist = "0"
       
        # construct objective function for current user using current price guideline 
        exec("lp_user%s += %s"%(user_id,user_function))
        
        # reset objective function
        user_function = "0"
        
        # calculate scheduling results for current user using current price guideline 
        exec("lp_user%s.solve()"%user_id)
        
        # add up scheduling results of current user by hours
        for hour in range(0,24):
            for task in task_by_hour[user_id-1][hour]:
                energy_usage[hour] += task.varValue
    
    # plot sum of scheduling results of all 5 users by hours using current price guideline
    plot_schedule(energy_usage,price[0])
    energy_usage = [0]*24



# In[ ]:





# In[ ]:




