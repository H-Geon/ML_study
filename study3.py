#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


baby_names=pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/US_Baby_Names/US_Baby_Names_right.csv')
baby_names.info()


# In[3]:


baby_names.head(10)


# In[4]:


del baby_names['Unnamed: 0']


# In[5]:


del baby_names['Id']


# In[6]:


baby_names.head()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# In[8]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')


# In[9]:


x=chipo.item_name
letter_counts=Counter(x)
df=pd.DataFrame.from_dict(letter_counts,orient='index')
df=df[0].sort_values(ascending=True)[45:50]
df.plot(kind='bar')


# In[10]:


chipo.item_price=[float(value[1:-1])for value in chipo.item_price]
orders=chipo.groupby('order_id').sum()


# In[11]:


orders=chipo.groupby('order_id').sum()


# In[12]:


plt.scatter(x=orders.item_price,y=orders.quantity,s=50,c='green')


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


path = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv'

online_rt = pd.read_csv(path, encoding = 'latin1')

online_rt.head()


# In[15]:


countries=online_rt.groupby('Country').sum()


# In[16]:


countries=countries.sort_values(by='Quantity',ascending=False)[1:11]


# In[17]:


countries['Quantity'].plot(kind='bar')


# In[18]:


online_rt=online_rt[online_rt.Quantity>0]
online_rt.head()


# In[19]:


customers=online_rt.groupby(['CustomerID','Country']).sum()


# In[20]:


raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
            'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'], 
            'female': [0, 1, 1, 0, 1],
            'age': [42, 52, 36, 24, 73], 
            'preTestScore': [4, 24, 31, 2, 3],
            'postTestScore': [25, 94, 57, 62, 70]}

df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'female', 'preTestScore', 'postTestScore'])

df


# In[21]:


plt.scatter(df.preTestScore,df.postTestScore,s=df.age)


# In[22]:


raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']                        
            }


# In[23]:


pokemon=pd.DataFrame(raw_data)


# In[24]:


pokemon=pokemon[['name','type','hp','evolution','pokedex']]
pokemon


# In[25]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine = pd.read_csv(url)

wine.head()


# In[26]:


wine.drop(wine.columns[[0,3,6,8,10,12,13]],axis=1,inplace=True)


# In[27]:


wine.head()


# In[28]:


wine.columns = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'magnesium', 'flavanoids', 'proanthocyanins', 'hue']
wine.head()


# In[29]:


wine.iloc[0:3,0]=np.nan


# In[30]:


wine.head()


# In[31]:


wine=wine.dropna(axis=0,how='any')
wine.head()


# In[32]:


mask=wine.alcohol.notnull()


# In[33]:


wine.alcohol[mask]


# In[34]:


wine=wine.reset_index(drop=True)
wine.head()

