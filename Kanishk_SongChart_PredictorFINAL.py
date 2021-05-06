#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import RidgeCV, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold


# In[34]:


df1 = pd.read_csv('C://Users//kkama//Downloads//top2018.csv', encoding="ISO-8859-1")
df1 = df1.drop(['time_signature', 'name'],axis='columns')
df1


# In[35]:


dummies = pd.get_dummies(df1.artists)
dummies


# In[36]:


merged = pd.concat([df1, dummies1], axis = 'columns')
df = merged.drop('artists', axis = 'columns')
df


# In[37]:


df.info()


# In[38]:


df.describe()


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=20, figsize=(20,20))
plt.show()


# In[40]:


df_copy = df.copy();


# In[41]:


df_train = df_copy.sample(frac=0.80, random_state=0)
df_train


# In[42]:


df_test = df_copy.drop(df_train.index)
df_test


# In[43]:


X=df_train.drop(['id'],axis='columns')
X


# In[44]:


Y=df_train.id
Y


# In[47]:


model = RandomForestRegressor(n_estimators = 1000)


# In[48]:


model.fit(X, Y)


# In[49]:


score = model.score(X, Y)
score


# In[50]:


df_test


# In[51]:


df_test_id=df_test.id
df_test_id


# In[52]:


df_final_test=df_test.drop(['id'], axis = 'columns')
df_final_test


# In[53]:


X_test_id = df_final_test
X_test_id


# In[54]:


Y_test_id = model.predict(X_test_id)
Y_test_id


# In[55]:


df_test_id_array = df_test_id.to_numpy(dtype=None, copy=False)
df_test_id_array


# In[56]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(df_test_id, Y_test_id))
rms


# In[57]:


plt.scatter(df_test_id_array,Y_test_id)   
plt.plot(df_test_id_array, df_test_id_array, color = 'red') 
plt
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')


# HOW TO INPUT: danceability	energy	key	loudness	mode	speechiness	acousticness	instrumentalness	liveness	valence	tempo	duration_ms

# In[59]:


model.predict([[ 0.727, 0.892, 1, -2.384, 1, 0.777, 0.431, 0.254, 0.226, 0.231, 90, 198973]])


# In[ ]:




