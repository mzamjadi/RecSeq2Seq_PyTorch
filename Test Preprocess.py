#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import unicode_literals, print_function, division
from numpy import *
import numpy as np

import pandas as pd
import datetime as dt
import csv
import torch
import pickle
from random import randint
import math

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


PATH_TO_ORIGINAL_DATA = '/home/mamjad2/trivagodata/'

#Preprocess the test data
test = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'test.csv')

#Subset data with not null itemID refrence of item ID
dftest1=test[(test['action_type'] != 'change of sort order') & (test['action_type'] !='filter selection') & (test['action_type'] !='search for destination')& (test['action_type'] !='search for poi') & (test['action_type'] !='clickout item') & (test['reference'].notnull())]
#Subset data with null refrence only for Clickouts
dftest2=test[(test['action_type'] == 'clickout item') & (test['reference'].isnull())] 

#Test data
dftest=dftest1.append(dftest2, ignore_index=True)


# In[ ]:


tstnp=[]
count=0
abnorm=[]
for x in set(dftest['user_id']):
    tstusr=dftest[dftest['user_id'] == x]
    for y in set(tstusr['session_id']):
        count += 1
        tstusr_session= tstusr[tstusr['session_id'] == y]
        tstusr_session=tstusr_session.sort_values('step')
        temp=tstusr_session['reference']#Note:This sequence of interacted items can have repetetive item even consequetively! 
        
        if len(temp)==1 and temp.isna().any()==True:
            #Sort impressions based on their probabilities
            continue
            
        elif len(list(np.where(tstusr_session['reference'].isna())))==1 and len(temp)-1 in np.where(tstusr_session['reference'].isna()):
            #temp=temp[:-1]
            #DO prdiction for seq
            continue
                
        elif tstusr_session['reference'].isna().any()==False:
            #Skip such rows
            continue
        else:
            abnorm.append(temp)
            with open('outfile', 'wb') as fp:
                pickle.dump(abnorm, fp)
            
        
        #datanp.append([NaNidx if math.isnan(x) else int(x) for x in train])
#print(counter)

#tstnp= sorted(tstnp, key=len, reverse=True)
#pickle.dump(tstnp, open('tstemp.pkle', 'wb'))

#np.savetxt('temp.txt', datanp)
#data=torch.from_numpy(np.array(datanp))


# In[ ]:




