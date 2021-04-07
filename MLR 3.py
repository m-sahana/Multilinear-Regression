#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:


corolla=pd.read_csv("D:\\projects\\Multi linear regression\\ToyotaCorolla.csv",encoding= 'unicode_escape')


# In[17]:


corolla.head(3)


# In[33]:


#as per given condition in the problem statement creating new dataframe considering specific conditions
data = pd.DataFrame(corolla, columns = ["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]) 


# In[34]:


data.head(3)


# In[35]:


data.corr()


# In[36]:


data.describe()


# In[37]:


import seaborn as sns


# In[38]:


sns.pairplot(data)


# In[39]:


data.isnull().sum()


# In[40]:


data.boxplot()


# In[41]:


data.hist()


# In[42]:


from sklearn.linear_model import LinearRegression


# In[43]:


lr1=LinearRegression()


# In[45]:


lr1.fit(data.iloc[:,1:9],data.Price)


# In[46]:


lr1.coef_


# In[47]:


lr1.intercept_


# In[49]:


lr1.score(data.iloc[:,1:9],data.Price)


# In[50]:


pred1=lr1.predict(data.iloc[:,1:9])


# In[53]:


np.sqrt(np.mean(pred1-data.Price)**2)


# In[54]:


plt.scatter(x=pred1,y=(pred1-data.Price));
plt.xlabel("Fitted");
plt.ylabel("Residuals");
plt.hlines(y=0,xmin=0,xmax=60)


# In[55]:


plt.hist(pred1-data.Price)


# In[56]:


plt.scatter(x=pred1,y=data.Price);
plt.xlabel("Predicted");
plt.ylabel("Actual");


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


train,test=train_test_split(data,test_size=0.2)


# In[59]:


from sklearn.linear_model import Ridge


# In[60]:


rm1=Ridge(alpha=0.4,normalize=True) 


# In[61]:


rm1.fit(train.iloc[:,1:9],train.Price)


# In[62]:


rm1.coef_


# In[63]:


rm1.intercept_


# In[64]:


pred_rm1=rm1.predict(train.iloc[:,1:9])


# In[65]:


rm1.score(train.iloc[:,1:11],train.Price)


# In[66]:


np.sqrt(np.mean(pred_rm1-train.Price)**2)


# In[68]:


train_rm1=[]
test_rm1=[]
R_sqrd=[]
alphas=np.arange(0,100,0.05)


# In[69]:


for i in alphas:
    RM=Ridge(alpha=i,normalize=True)
    RM.fit(train.iloc[:,1:9],train.Price)
    R_sqrd.append(RM.score(train.iloc[:,1:9],train.Price))
    train_rm1.append(np.sqrt(np.mean((RM.predict(train.iloc[:,1:9])-train.Price)**2)))
    test_rm1.append(np.sqrt(np.mean((RM.predict(test.iloc[:,1:9])-test.Price)**2)))


# In[70]:


#Alpha vs r-squared value
plt.scatter(x=alphas,y=R_sqrd);
plt.xlabel("Alphas");
plt.ylabel("R_sqrd");


# In[71]:


#alpha vs train_rm1
plt.scatter(x=alphas,y=train_rm1);
plt.xlabel("alphas");
plt.ylabel("train_rm1");


# In[72]:


#alphas vs test rm1
plt.scatter(x=alphas,y=test_rm1);
plt.xlabel("alphas");
plt.ylabel("test_rm1");


# In[73]:


from sklearn.linear_model import Lasso


# In[74]:


Lasso_model1=Lasso(alpha=0.01,normalize=True)


# In[75]:


Lasso_model1.fit(train.iloc[:,1:9],train.Price)


# In[76]:


Lasso_model1.coef_


# In[77]:


Lasso_model1.intercept_


# In[78]:


pred_LassoM1=Lasso_model1.predict(train.iloc[:,1:9])


# In[79]:


Lasso_model1.score(train.iloc[:,1:9],train.Price)


# In[80]:


np.sqrt(np.mean(pred_LassoM1-train.Price))


# In[82]:


train_rm2=[]
test_rm2=[]
R_sqrd_2=[]
alpha=np.arange(0,30,0.05)


# In[83]:


for i in alpha:
    RM=Ridge(alpha=i,normalize=True)
    RM.fit(train.iloc[:,1:9],train.Price)
    R_sqrd_2.append(RM.score(train.iloc[:,1:9],train.Price))
    train_rm2.append(np.sqrt(np.mean((RM.predict(train.iloc[:,1:9])-train.Price)**2)))
    test_rm2.append(np.sqrt(np.mean((RM.predict(test.iloc[:,1:9])-test.Price)**2)))


# In[84]:


plt.scatter(x=alpha,y=R_sqrd_2);
plt.xlabel("Alphas");
plt.ylabel("R_sqrd");


# In[85]:


plt.scatter(x=alpha,y=train_rm2);
plt.xlabel("Alphas");
plt.ylabel("train_rm2");


# In[86]:


plt.scatter(x=alpha,y=test_rm2);
plt.xlabel("Alphas");
plt.ylabel("test_rm2");


# In[ ]:




