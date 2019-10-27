
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pickle
import time
import csv
import cProfile


# In[2]:


def get_data():
    train = pd.read_csv('train.csv') #read data
    #train['action_type'].unique() #unique values in action_type column

    ##Subseting the train data into the clickouts only
    dftrain=train[train['action_type'] == 'clickout item'] 

    ##Computing the probability of clicking each item
    freq=dftrain['reference'].value_counts() #Counts the frequencies

    prob=freq/len(dftrain['reference']) #compute the probablitie
    probdic=dict(prob) #Store items as keys in the dict

    test=pd.read_csv('test.csv')
    #test.head(5) #display first 5 rows

    # test['action_type'].unique() #unique values in action_type column

    ##Subseting the data into the clickouts and null only
    dftest=test[(test['action_type'] == 'clickout item') & (test['reference'].isnull())] 
    impdict=dict(dftest['impressions'])
    dfsubmit=dftest.loc[:,['user_id','session_id','timestamp', 'step', 'impressions']]
    dfsubmit.columns=['user_id','session_id','timestamp', 'step', 'item_recommendations']

    return probdic,impdict,dfsubmit


# In[3]:


def Scorenew(impressions_string, probdic):
    
    implist= impressions_string.split("|")
    probimp=[]
    for j in implist:
        if j in probdic:
            probimp.append(probdic[j])
        else:
            probimp.append(0)


    temp = sorted(zip(probimp,implist), reverse = True)

    return ' '.join([x for _,x in temp])


# In[4]:


[probdic,impdict,dfsubmit] = pickle.load(open('processed.pkl','rb'))


# In[ ]:


if __name__=='__main__':

    # probdic,impdict,dfsubmit = get_data()
    # pickle.dump([probdic,impdict,dfsubmit],open('processed.pkl','wb'))

    [probdic,impdict,dfsubmit] = pickle.load(open('processed.pkl','rb'))

    count = 0
    for x in impdict:
        start = time.time()
        dfsubmit['item_recommendations'][x] = Scorenew(impdict[x],probdic)
        count += 1
        if count % 10000 == 0:
            dfsubmit.to_csv("./submit.csv", sep=',',index=False)
            print(count)
            

