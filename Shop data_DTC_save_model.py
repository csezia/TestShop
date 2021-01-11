#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd


# In[75]:


df=pd.read_csv('shop data.csv')


# In[76]:


df


# In[77]:


x=df.drop(['buys'], axis=1)


# In[78]:


x


# In[79]:


y=df['buys']


# In[80]:


y


# In[81]:


from sklearn.preprocessing import LabelEncoder


# In[82]:


le_x=LabelEncoder


# In[83]:


x=x.apply(LabelEncoder().fit_transform)
x


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.25, random_state=0)


# In[86]:


xtest


# In[87]:


from sklearn.linear_model import LogisticRegression


# In[88]:


model=LogisticRegression()


# In[89]:


model.fit(xtrain,ytrain)


# In[90]:


model.predict(xtest)


# In[91]:


model.predict([[1,0,1,0]])


# In[92]:


model.score(xtest,ytest)


# In[93]:


model.predict_proba(xtest)


# In[94]:


import numpy as np


# In[95]:


xinputt=np.array([1,0,0,0])


# In[96]:


y_predictt=model.predict([xinputt])


# In[97]:


y_predictt


# In[98]:


model.score(x,y)


# In[99]:


from sklearn.tree import DecisionTreeClassifier


# In[100]:


dtf=DecisionTreeClassifier()


# In[101]:


dtf.fit(xtrain,ytrain)


# In[102]:


import numpy as np


# In[103]:


xinput=np.array([1,1,0,0])


# In[104]:


y_predict=dtf.predict([xinput])


# In[105]:


y_predict


# In[106]:


model.score(x,y)


# In[107]:


import pickle


# In[108]:


with open('data.pkl','wb') as file:
    pickle.dump(model,file)


# In[109]:


with open('data.pkl','rb') as file:
    mp = pickle.load(file)


# In[110]:


y_predict


# In[ ]:




