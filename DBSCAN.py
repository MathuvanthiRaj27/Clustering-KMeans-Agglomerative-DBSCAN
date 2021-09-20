#!/usr/bin/env python
# coding: utf-8

# In[54]:


''' Importing the libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from collections import Counter
import pickle


# In[55]:


''' Reading the csv file '''
data = pd.read_csv("jewellery.csv") 


# In[56]:


''' Printing the 1st 5 rows '''
data.head()


# In[57]:


''' Printing the datatypes '''
data.dtypes


# In[58]:


''' Printing the shape of the dataset'''
data.shape


# In[59]:


''' Checking for null values '''
data.isnull().sum()


# In[60]:


df= pd.DataFrame(data['Age'])
df['SpendingScore']=data['SpendingScore']
df.head()


# In[61]:


'''Implementing DBSCAN'''

dbscan = DBSCAN(eps = 3, min_samples = 4)
db = dbscan.fit_predict(df)
data["cluster"] = db
labels = dbscan.labels_
print(labels)


# In[62]:


''' Finding the clusters '''

n_clusters = len(set(labels))
print(n_clusters)


# In[63]:


print(Counter(dbscan.labels_))


# In[64]:


''' Silhoutte score'''

db_score = silhouette_score(df, dbscan.labels_)
print('Silhouette Score: %.3f' % db_score)


# In[66]:


pickle.dump(dbscan,open("dbscan.pkl","wb"))


# In[ ]:




