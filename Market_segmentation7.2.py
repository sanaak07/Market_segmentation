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
df = pd.read_csv("Mall_Customers.csv")
df.head()


# In[3]:


#Checking the size of our data
df.shape


# In[4]:


#Changing the name of some columns
df = df.rename(columns={'Annual Income (k$)': 'Annual_income', 'Spending Score (1-100)': 'Spending_score'})


# In[5]:


#Looking for null values
df.isna().sum()


# In[6]:


#Checking datatypes
df.info()


# In[7]:


#Replacing objects for numerical values
df['Genre'].replace(['Female','Male'], [0,1],inplace=True)


# In[8]:


#Checking values have been replaced properly
df.Genre


# In[9]:


#Density estimation of values using distplot
plt.figure(1 , figsize = (15 , 6))
feature_list = ['Age','Annual_income', "Spending_score"]
feature_listt = ['Age','Annual_income', "Spending_score"]
pos = 1 
for i in feature_list:
    plt.subplot(1 , 3 , pos)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.distplot(df[i], bins=20, kde = True)
    pos = pos + 1
plt.show()


# In[10]:


#Count and plot gender
sns.countplot(y = 'Genre', data = df, palette="husl", hue = "Genre")
df["Genre"].value_counts()


# In[11]:


#Pairplot with variables we want to study
sns.pairplot(df, vars=["Age", "Annual_income", "Spending_score"],  kind ="reg", hue = "Genre", palette="husl", markers = ['o','D'])


# In[12]:


#age and annual income
sns.lmplot(x = "Age", y = "Annual_income", data = df, hue = "Genre")


# In[13]:


#spending score and annual income
sns.lmplot(x = "Annual_income", y = "Spending_score", data = df, hue = "Genre")


# In[14]:


#age and spending score
sns.lmplot(x = "Age", y = "Spending_score", data = df, hue = "Genre")


# # Selecting number of clusters

# In[15]:


#Creating values for the elbow
X = df.loc[:,["Age", "Annual_income", "Spending_score"]]
inertia = []
k = range(1,20)
for i in k:
    means_k = KMeans(n_clusters=i, random_state=0)
    means_k.fit(X)
    inertia.append(means_k.inertia_)


# In[16]:


#Plotting the elbow
plt.plot(k , inertia , 'bo-')
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


# # Clustering

# In[17]:


#Training kmeans with 5 clusters
means_k = KMeans(n_clusters=5, random_state=0)
means_k.fit(X)
labels = means_k.labels_
centroids = means_k.cluster_centers_


# In[18]:


#Create a 3d plot to view the data separation made by Kmeans
trace1 = go.Scatter3d(
    x= X['Spending_score'],
    y= X['Annual_income'],
    z= X['Age'],
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
            xaxis = dict(title  = 'Spending_score'),
            yaxis = dict(title  = 'Annual_income'),
            zaxis = dict(title  = 'Age')
        )
)
fig = go.Figure(data=trace1, layout=layout)
py.offline.iplot(fig)


# In[23]:





# In[ ]:




