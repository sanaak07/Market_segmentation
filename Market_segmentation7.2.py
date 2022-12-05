#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#We read the csv and print the first 5 rows
df = pd.read_csv("Mac.donalds.csv")
df.head()


# In[3]:


#Checking the size of our data
df.shape


# In[4]:


#label encoding
df.head()


# In[5]:


#dropping convienient
df.drop(['convenient'], axis=1, inplace=True)


# In[6]:


#dropping spicy
df.drop(['spicy'], axis=1, inplace=True)


# In[7]:


#dropping fattening
df.drop(['fattening'], axis=1, inplace=True)


# In[8]:


#dropping greasy
df.drop(['greasy'], axis=1, inplace=True)


# In[9]:


#dropping fast
df.drop(['fast'], axis=1, inplace=True)


# In[10]:


#dropping cheap
df.drop(['cheap'], axis=1, inplace=True)


# In[11]:


#dropping like
df.drop(['Like'], axis=1, inplace=True)


# In[12]:


#dropping visiting frequency
df.drop(['VisitFrequency'], axis=1, inplace=True)


# In[13]:


#label encoding
df.head()


# In[14]:


#Replacing objects for numerical values
df['tasty'].replace(['No','Yes'], [0,1],inplace=True)
df['expensive'].replace(['No','Yes'], [0,1],inplace=True)
df['healthy'].replace(['No','Yes'], [0,1],inplace=True)
df['disgusting'].replace(['No','Yes'], [0,1],inplace=True)
df['Gender'].replace(['Male','Female'], [0,1],inplace=True)


# In[15]:


#label encoding
df.head()


# In[16]:


#Density estimation of values using distplot
plt.figure(1 , figsize = (15 , 6))
feature_list = ['tasty','expensive', "healthy"]
feature_listt = ['tasty','expensive', "healthy"]
pos = 1 
for i in feature_list:
    plt.subplot(1 , 3 , pos)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.distplot(df[i], bins=20, kde = True)
    pos = pos + 1
plt.show()


# In[17]:


#Count and plot gender
sns.countplot(y = 'Gender', data = df, palette="husl", hue = "Gender")
df["Gender"].value_counts()


# In[18]:


#Pairplot with variables we want to study
sns.pairplot(df, vars=["tasty", "expensive", "healthy"],  kind ="reg", hue = "Gender", palette="husl", markers = ['o','D'])


# In[19]:


#healthy and age
sns.lmplot(x = "healthy", y = "Age", data = df, hue = "Gender")


# In[20]:


#expensive and healthy
sns.lmplot(x = "expensive", y = "healthy", data = df, hue = "Gender")


# In[21]:


#tasty and expensive
sns.lmplot(x = "tasty", y = "expensive", data = df, hue = "Gender")


# # Selecting number of clusters

# In[22]:


#Creating values for the elbow
X = df.loc[:,["healthy", "Age", "Gender"]]
inertia = []
k = range(1,20)
for i in k:
    means_k = KMeans(n_clusters=i, random_state=0)
    means_k.fit(X)
    inertia.append(means_k.inertia_)


# In[23]:


#Plotting the elbow
plt.plot(k , inertia , 'bo-')
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# # Clustering

# In[24]:


#Training kmeans with 5 clusters
means_k = KMeans(n_clusters=5, random_state=0)
means_k.fit(X)
labels = means_k.labels_
centroids = means_k.cluster_centers_


# In[25]:


#Create a 3d plot to view the data separation made by Kmeans
trace1 = go.Scatter3d(
    x= X['healthy'],
    y= X['Age'],
    z= X['Gender'],
    mode='markers',
     marker=dict(
        color = labels, 
        size= 10,
        line=dict(
            color= labels,
        ),
        opacity = 0.9
     )
)
layout = go.Layout(
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'healthy'),
            yaxis = dict(title  = 'Age'),
            zaxis = dict(title  = 'Gender')
        )
)
fig = go.Figure(data=trace1, layout=layout)
py.offline.iplot(fig)


# In[ ]:




