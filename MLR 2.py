#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:


data=pd.read_csv("D:\projects\Multi linear regression\Computer_Data.csv")


# In[17]:


data.rename(columns={'Unnamed: 0':'Sln'},inplace=True)
data.head()
data.cd.replace(('yes', 'no'), (1, 0), inplace=True)
data.multi.replace(('yes', 'no'), (1, 0), inplace=True)
data.premium.replace(('yes', 'no'), (1, 0), inplace=True)


# In[65]:


data.corr()
#data.iloc[:,2:11]


# In[66]:


data.describe()


# In[67]:


import seaborn as sns


# In[68]:


sns.pairplot(data)


# In[69]:


data.isnull().sum()


# In[70]:


data.boxplot()


# In[71]:


data.hist()


# In[72]:


from sklearn.linear_model import LinearRegression


# In[80]:


lr1=LinearRegression()


# In[81]:


lr1.fit(data.iloc[:,1:11],data.price)


# In[82]:


lr1.coef_


# In[83]:


lr1.intercept_  #beta0


# In[84]:


lr1.score(data.iloc[:,1:11],data.price)


# In[85]:


pred1=lr1.predict(data.iloc[:,1:11])


# In[86]:


np.sqrt(np.mean(pred1-data.price)**2)


# In[87]:


plt.scatter(x=pred1,y=(pred1-data.price));
plt.xlabel("Fitted");
plt.ylabel("Residuals");
plt.hlines(y=0,xmin=0,xmax=60)


# In[88]:


plt.hist(pred1-data.price)


# In[89]:


plt.scatter(x=pred1,y=data.price);
plt.xlabel("Predicted");
plt.ylabel("Actual");


# In[90]:


np.corrcoef(data.speed,data.price)  #30%


# In[91]:


np.corrcoef(data.hd,data.price)  #43%


# In[92]:


np.corrcoef(data.ram,data.price)   #62%


# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


train,test=train_test_split(data,test_size=0.2)


# In[95]:


from sklearn.linear_model import Ridge


# In[96]:


rm1=Ridge(alpha=0.4,normalize=True)                  


# In[97]:


rm1.fit(train.iloc[:,1:11],train.price)


# In[98]:


rm1.coef_


# In[99]:


rm1.intercept_


# In[100]:


rm1.alpha


# In[101]:


pred_rm1=rm1.predict(train.iloc[:,1:11])


# In[102]:


rm1.score(train.iloc[:,1:11],train.price)


# In[132]:


np.sqrt(np.mean(pred_rm1-train.price)**2)


# In[133]:


train_rm1=[]          #rm is nothing but rmse= root mean square error


# In[134]:


test_rm1=[]


# In[135]:


R_sqrd=[]


# In[136]:


alphas=np.arange(0,100,0.05)


# In[137]:


for i in alphas:
    RM=Ridge(alpha=i,normalize=True)
    RM.fit(train.iloc[:,1:11],train.price)
    R_sqrd.append(RM.score(train.iloc[:,1:11],train.price))
    train_rm1.append(np.sqrt(np.mean((RM.predict(train.iloc[:,1:11])-train.price)**2)))
    test_rm1.append(np.sqrt(np.mean((RM.predict(test.iloc[:,1:11])-test.price)**2)))


# In[138]:


#Alpha vs r-squared value
plt.scatter(x=alphas,y=R_sqrd);
plt.xlabel("Alphas");
plt.ylabel("R_sqrd");


# In[139]:


#alpha vs train_rm1
plt.scatter(x=alphas,y=train_rm1);
plt.xlabel("alphas");
plt.ylabel("train_rm1");


# In[140]:


#alphas vs test rm1
plt.scatter(x=alphas,y=test_rm1);
plt.xlabel("alphas");
plt.ylabel("test_rm1");


# In[141]:


from sklearn.linear_model import Lasso


# In[142]:


Lasso_model1=Lasso(alpha=0.01,normalize=True)


# In[143]:


Lasso_model1.fit(train.iloc[:,1:11],train.price)


# In[144]:


Lasso_model1.coef_


# In[145]:


Lasso_model1.intercept_


# In[146]:


pred_LassoM1=Lasso_model1.predict(train.iloc[:,1:11])


# In[147]:


Lasso_model1.score(train.iloc[:,1:11],train.price)


# In[148]:


np.sqrt(np.mean(pred_LassoM1-train.price))


# In[149]:


train_rm2=[]


# In[150]:


test_rm2=[]


# In[151]:


R_sqrd_2=[]


# In[152]:


alpha=np.arange(0,30,0.05)


# In[153]:


for i in alpha:
    RM=Ridge(alpha=i,normalize=True)
    RM.fit(train.iloc[:,1:11],train.price)
    R_sqrd_2.append(RM.score(train.iloc[:,1:11],train.price))
    train_rm2.append(np.sqrt(np.mean((RM.predict(train.iloc[:,1:11])-train.price)**2)))
    test_rm2.append(np.sqrt(np.mean((RM.predict(test.iloc[:,1:11])-test.price)**2)))


# In[154]:


plt.scatter(x=alpha,y=R_sqrd_2);
plt.xlabel("Alphas");
plt.ylabel("R_sqrd");


# In[155]:


plt.scatter(x=alpha,y=train_rm2);
plt.xlabel("Alphas");
plt.ylabel("train_rm2");


# In[156]:


plt.scatter(x=alpha,y=test_rm2);
plt.xlabel("Alphas");
plt.ylabel("test_rm2");


# In[ ]:




