#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#import dataset
dataset=pd.read_csv('Tree_Species.csv')
dataset


# In[4]:


X=dataset.iloc[:, [2,3]].values


# In[5]:


#import kmeans from sklearn
from sklearn.cluster import KMeans


# In[6]:


#plot elbow method graph
wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters= i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('wcss')
plt.show


# In[8]:


#gunakan 4 cluster sesuai dengan elbow method di atas
kmeans=KMeans(n_clusters= 4, init='k-means++', max_iter=300, n_init=10, random_state=0)


# In[9]:


#predict the cluster
y_kmeans = kmeans.fit_predict(X)


# In[12]:


#visualisasi 4 cluster
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label = 'Tree 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label = 'Tree 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label = 'Tree 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label = 'Tree 4')

#menentukan centroid untuk setiap cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')
plt.title('Cluster of Tree')
plt.xlabel('Widht')
plt.ylabel('Height')
plt.legend()
plt.show()


# In[ ]:




