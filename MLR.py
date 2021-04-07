#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


startup=pd.read_csv("D:\\projects\\Multi linear regression\\50_Startups.csv")
startup.columns=['RD','A','MS','S','P']


# In[5]:


startup.head(5)


# In[6]:


startup.corr() 
#highly correlated between r&d and profit, marketing spend and profit


# In[7]:


import seaborn as sns


# In[8]:


sns.pairplot(startup)


# In[9]:


import statsmodels.formula.api as smf


# In[15]:


model = smf.ols('P~RD+A+MS', data=startup).fit()
model.params
model.summary()


# In[17]:


m1 = smf.ols('P~A', data=startup).fit()
m1.summary()


# In[18]:


m2 = smf.ols('P~MS', data=startup).fit()
m2.summary()


# In[19]:


m3 = smf.ols('P~A+MS', data=startup).fit()
m3.summary()


# In[21]:


import statsmodels.api as sm
sm.graphics.influence_plot(model)


# In[22]:


startup_new=startup.drop(startup.index[[45,49]],axis=0)


# In[24]:


model_new=smf.ols('P~RD+A+MS',data=startup_new).fit()
model_new.params
model_new.summary()


# In[25]:


print(model_new.conf_int(0.001))


# In[ ]:




