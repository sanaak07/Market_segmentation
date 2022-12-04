#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# For interactive visualizations
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff


# In[3]:


# importing the dataset
data = pd.read_csv('Mall_customers.csv')
data.shape


# In[4]:


#see top 5 dataset
data.head()


# In[5]:


#see last 5 datasets
data.tail()


# In[6]:


data.info()


# In[7]:


data.describe()


# # Data Visualisation

# In[10]:


import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize']=(14,6)

plt.subplot(1,2,1)
sns.set(style = 'whitegrid')
sns.distplot(data['Annual Income (k$)'])
plt.title('Distribution of Annual Income',fontsize=15)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.set(style = 'whitegrid')
sns.distplot(data['Age'],color = 'orange')
plt.title('Distribution of Age', fontsize=15)
plt.xlabel('Range of Age')
plt.ylabel('Count')
plt.show()


# In[13]:


plt.rcParams['figure.figsize'] = (14,8)
sns.countplot(data['Age'],palette = 'hsv')
plt.title('Distribution of Age', fontsize = 15)
plt.show()


# In[14]:


plt.rcParams['figure.figsize'] = (14,8)
sns.countplot(data['Annual Income (k$)'], palette= 'rainbow')
plt.title('Distribution of Annual Income', fontsize = 15)
plt.show()


# In[15]:


sns.pairplot(data)
plt.title('Pairplot for the Data', fontsize = 15)
plt.show()


# In[17]:


x=data.iloc[:,[3,4]].values

print(x.shape)


# # k-means Clustering

# In[18]:


from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    km=KMeans(n_clusters=i, init = 'k-means++', max_iter = 300,n_init=10, random_state = 0 )
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No.of Clusters')
plt.ylabel('wcss')
plt.show()


# In[19]:


km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'green', label = 'miser')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')
plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200, c = 'blue' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[20]:


#use of dendograms

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 15)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()


# In[21]:


#vislualise clusters of heirarchical clustering

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'green', label = 'miser')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200, c = 'blue' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('Hierarchial Clustering', fontsize = 15)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




