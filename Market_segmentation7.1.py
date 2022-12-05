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


# In[5]:


# importing the dataset
df = pd.read_csv('Mac.donalds.csv')
df.shape


# In[7]:


#label encoding
df.head()


# In[40]:


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


# In[41]:


#Replacing objects for numerical values
df['tasty'].replace(['No','Yes'], [0,1],inplace=True)
df['expensive'].replace(['No','Yes'], [0,1],inplace=True)
df['healthy'].replace(['No','Yes'], [0,1],inplace=True)
df['disgusting'].replace(['No','Yes'], [0,1],inplace=True)
df['Gender'].replace(['Male','Female'], [0,1],inplace=True)


# In[42]:


#label encoding
df.head()


# In[13]:


#label encoding
df.describe()


# In[14]:


df


# # Data Visualisation

# In[17]:


import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize']=(14,6)

plt.subplot(1,2,1)
sns.set(style = 'whitegrid')
sns.distplot(data['Age'])
plt.title('Distribution of Age',fontsize=15)
plt.xlabel('Range of Age')
plt.ylabel('Count')


# In[18]:


plt.rcParams['figure.figsize'] = (14,8)
sns.countplot(data['Age'],palette = 'hsv')
plt.title('Distribution of Age', fontsize = 15)
plt.show()


# In[20]:


plt.rcParams['figure.figsize'] = (14,8)
sns.countplot(data['Gender'], palette= 'rainbow')
plt.title('Distribution of Gender', fontsize = 15)
plt.show()


# In[21]:


sns.pairplot(data)
plt.title('Pairplot for the Data', fontsize = 15)
plt.show()

