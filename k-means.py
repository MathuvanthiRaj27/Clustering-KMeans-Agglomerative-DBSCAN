#!/usr/bin/env python
# coding: utf-8

# In[1]:


''' Importing the libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pickle


# In[2]:


''' Reading the csv file '''
data = pd.read_csv("jewellery.csv") 


# In[3]:


''' Printing the 1st 5 rows '''
data.head()


# In[4]:


''' Printing the datatypes '''
data.dtypes


# In[5]:


''' Printing the shape of the dataset'''
data.shape


# In[7]:


''' Checking for null values '''
data.isnull().sum()


# In[9]:


df= pd.DataFrame(data['Age'])
df['SpendingScore']=data['SpendingScore']
df.head()


# In[18]:


''' k-means elbow curve'''
    
sse=[]
no_of_cluster= []

for i in range(1,10): 
    km = KMeans(n_clusters=i)
    cluster_predicted = km.fit(df)
    no_of_cluster.append(i) 
    sse.append(cluster_predicted.inertia_)
    print(f'sse {i}: {cluster_predicted.inertia_}')
plt.figure(figsize=(10,5))
plt.plot(no_of_cluster, sse)


# In[19]:


'''Initializing K-means cluster as 3'''
km = KMeans(n_clusters=3)
print(km)


# In[20]:


''' Fitting and predicting the values'''
predict = km.fit_predict(df)
print(predict)


# In[21]:


''' Finding the silhouette score'''

score = silhouette_score(df, km.labels_, metric='euclidean')
print('Silhouette Score: %.3f' % score)


# In[22]:


'''Printing the centroids of each column'''

center = km.cluster_centers_
print(center)


# In[23]:


'''Visualizing the age and income with its respective clusters'''
df['cluster'] = predict
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age, df1.SpendingScore, color='black')
plt.scatter(df2.Age, df2.SpendingScore, color='red')
plt.scatter(df3.Age, df3.SpendingScore, color='green')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='yellow', marker='*', s=800)


# In[24]:


pickle.dump(km,open("kmeans.pkl","wb"))


# In[ ]:




