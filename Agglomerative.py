#!/usr/bin/env python
# coding: utf-8

# In[3]:


''' Importing the libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import pickle


# In[4]:


''' Reading the csv file '''
data = pd.read_csv("jewellery.csv") 


# In[5]:


''' Printing the 1st 5 rows '''
data.head()


# In[6]:


''' Printing the datatypes '''
data.dtypes


# In[7]:


''' Printing the shape of the dataset'''
data.shape


# In[9]:


''' Checking for null values '''
data.isnull().sum()


# In[10]:


df= pd.DataFrame(data['Age'])
df['SpendingScore']=data['SpendingScore']
df.head()


# In[13]:


'''Agglomerative Clustering '''

ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
pre = ac.fit_predict(df)
print(pre)


# In[14]:


''' Silhoutte score'''

ac_score = silhouette_score(df, ac.labels_)
print('Silhouette Score: %.3f' % ac_score)


# In[15]:


'''Printing the count of each clusters'''

print(Counter(ac.labels_))


# In[17]:


pickle.dump(ac,open("agglomerative.pkl","wb"))

