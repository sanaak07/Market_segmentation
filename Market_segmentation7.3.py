#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np   
import seaborn as sns 
from datetime import datetime 
import plotly.express as px 
import plotly.figure_factory as ff 

import matplotlib.pyplot as plt 

### Clustering libraries 
from sklearn.cluster import KMeans   

### Clustering Metrics 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import davies_bouldin_score  
from sklearn.metrics import calinski_harabasz_score  

import warnings
warnings.filterwarnings('ignore')


# In[2]:


def side_by_side(*objs, **kwds):
    from pandas.io.formats.printing import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print (adjoin(space, *reprs))
    print()
    return 


# In[3]:


# importing the dataset
df = pd.read_csv('Mac.donalds.csv')
df.shape


# In[4]:


#label encoding
df.head()


# In[6]:


#label encoding
df.columns


# In[7]:


#dropping convienient
df.drop(['yummy'], axis=1, inplace=True)
#dropping convienient
df.drop(['convenient'], axis=1, inplace=True)
#dropping spicy
df.drop(['spicy'], axis=1, inplace=True)
#dropping fattening
df.drop(['fattening'], axis=1, inplace=True)
#dropping greasy
df.drop(['greasy'], axis=1, inplace=True)
#dropping fast
df.drop(['fast'], axis=1, inplace=True)
#dropping cheap
df.drop(['cheap'], axis=1, inplace=True)
#dropping like
df.drop(['Like'], axis=1, inplace=True)
#dropping visiting frequency
df.drop(['VisitFrequency'], axis=1, inplace=True)

#label encoding
df.head()


# In[8]:


#Replacing objects for numerical values
df['tasty'].replace(['No','Yes'], [0,1],inplace=True)
df['expensive'].replace(['No','Yes'], [0,1],inplace=True)
df['healthy'].replace(['No','Yes'], [0,1],inplace=True)
df['disgusting'].replace(['No','Yes'], [0,1],inplace=True)
df['Gender'].replace(['Male','Female'], [0,1],inplace=True)


# In[9]:


#label encoding
df.head()


# In[10]:


#label encoding
df.describe()


# In[11]:


#visualising data by gender
male_customers = df[df.Gender == 0].shape[0]
female_customers = df[df.Gender == 1].shape[0]

px.pie(values=[male_customers, female_customers], names=['Male', 'Female'], title='Gender', width=600, height=400)


# In[12]:


#Visualising box plot
fig = px.box(df, y="Age", x='Gender', width=600, height=400)
fig.show() 


# In[13]:


#visualising histogram of distribution of age
fig = px.histogram(df, x="Age", color="Gender", marginal="rug", width=600, height=400)
fig.show()


# In[15]:


#distribution of gender
fig = px.histogram(df, x="Gender", color="Gender", marginal="rug", width=600, height=400)
fig.show()


# In[17]:


#distribution of tasty scores
fig = px.histogram(df, x="tasty", color="Gender", marginal="rug", width=600, height=400)
fig.show()


# In[18]:


fig,ax = plt.subplots(figsize=(6,4)) ## play with size 
fig.suptitle("Correlation between features", fontsize=16)
corrcoef = df.corr()
mask = np.array(corrcoef)
mask[np.tril_indices_from(mask)] = False
sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)
plt.show(); 


# In[19]:


get_ipython().system('pip install yellowbrick')


# # k means Clustering

# In[20]:


from yellowbrick.cluster import KElbowVisualizer 

print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=14, timings=False)
Elbow_M.fit(df)
Elbow_M.show() 


# # Fuzzy means clustering

# In[21]:


get_ipython().system(' pip install fuzzy-c-means')


# In[22]:


number_clusters = 5 


# In[23]:


from fcmeans import FCM 

fcm = FCM(n_clusters=number_clusters)
fcm.fit(df.values)

# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(df.values) 


# In[24]:


df['Fuzzy_cluster'] = fcm_labels


# In[25]:


df


# In[27]:


sns.relplot(x='Age', y='Gender', hue='Fuzzy_cluster', size='Gender', data=df, palette = "Paired") 


# In[28]:


sns.relplot(x='Age', y='Gender', hue='Fuzzy_cluster', data=df, palette = "Paired")


# # Evaluating Fuzzy C Means

# In[29]:


print("Silhouette score: {}".format(silhouette_score(df.drop(['Fuzzy_cluster'], axis=1), fcm_labels)))
print("Davies Bouldin score: {}".format(davies_bouldin_score(df.drop(['Fuzzy_cluster'], axis=1), fcm_labels)))
print("Calinski Harabasz score: {}".format(calinski_harabasz_score(df.drop(['Fuzzy_cluster'], axis=1), fcm_labels))) 


# # K means

# In[30]:


df.drop(['Fuzzy_cluster'], axis=1, inplace=True)


# In[31]:


# InterclusterDistance 
from yellowbrick.cluster import SilhouetteVisualizer  
from yellowbrick.cluster import InterclusterDistance   


model = KMeans(number_clusters)
visualizer = InterclusterDistance(model) 

visualizer.fit(df)    
visualizer.show()   


# # Silhoutte Visualiser

# In[32]:


model = KMeans(number_clusters, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(df)               # Fit the data to the visualizer
visualizer.show()                # Finalize and render the figure


# In[ ]:




