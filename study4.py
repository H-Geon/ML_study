#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns',100)


# In[8]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')


# In[9]:


trainset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')


# In[10]:


trainset.head()


# In[12]:


print('Train dataset(rows,cols):',trainset.shape,'\nTest dataset(rows,cols):',testset.shape)


# In[17]:



data =[]
for feature in trainset.columns:
    #Defininfg the role
    if feature == 'target':
        user = 'target'
    elif feature == 'id':
        use= 'id'
    else:
        use= 'input'
    
    #Defining the type
    if 'bin' in feature or feature == 'target':
        type = 'binary'
    elif 'cat' in feature or feature == 'id':
        type = 'categorical'
    elif trainset[feature].dtype == float or isinstance(trainset[feature].dtype,float):
        type = 'real'
    elif trainset[feature].dtype == int:
        type = 'integer'
    
    #Initialize preserve to True for all variables except for id
    preserve = True
    if feature == 'id':
        preserve = False

    #Defining the data type
    dtype = trainset[feature].dtype
    category = 'none'
    
    #Defining the category
    if 'ind' in feature:
        category = 'individual'
    elif 'reg' in feature:
        category = 'registration'
    elif 'car' in feature:
        category = 'car'
    elif 'calc' in feature:
        category = 'calculated'
    
    #Creating a Dict that contains all the metadata for the variable
    feature_dictionary = {
        'varname':feature,
        'use':use,
        'type':type,
        'preserve':preserve,
        'dtype':dtype,
        'category':category
    }
    data.append(feature_dictionary)

metadata=pd.DataFrame(data,columns=['varname','use','type','preserve','dtype','category'])
metadata.set_index('varname',inplace=True)
metadata


# In[18]:


metadata[(metadata.type =='categorical')&(metadata.preserve)].index


# In[19]:


pd.DataFrame({'count':metadata.groupby(['category'])['category'].size()}).reset_index()


# In[20]:


pd.DataFrame({'count':metadata.groupby(['use','type'])['use'].size()}).reset_index()


# In[24]:


plt.figure()
fig,ax = plt.subplots(figsize=(6,6))
x=trainset['target'].value_counts().index.values
y=trainset['target'].value_counts().values
sns.barplot(ax=ax,x=x,y=y)
plt.ylabel('Number of values',fontsize=12)
plt.xlabel('Target value',fontsize=12)
plt.tick_params(axis='both',which='major',labelsize=12)
plt.show();

