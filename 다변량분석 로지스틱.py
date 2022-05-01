#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


from IPython.display import display


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# 기계학습 모델
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


# In[6]:


data = pd.read_csv('Parkinson.csv')
data.head()


# In[7]:


data.shape


# In[8]:


data = data.drop(['name'], axis=1)
data.head()
#카데고리형 변수가 없으므로, 더미화 시키지 않았다


# In[9]:


data.shape


# In[10]:


plt.figure(figsize = (8,8))
sns.pairplot(data[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'status']], hue="status")
plt.show()
#0 for healthy and 1 for PD
#범주형 변수로는 status를 사용, 그 외 변수는 3가지를 사용했다.


# In[11]:


display(data['status'].value_counts())


# In[12]:


X = data.drop('status', axis=1)
y = data['status']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=20190625, stratify=y)
print('클래스별 데이터 개수: Training 데이터  ')
display(y_train.value_counts())
print('클래스별 데이터 개수: Testing')
display(y_test.value_counts())


# In[13]:


model = sm.Logit(y_train, X_train)
model_fitted = model.fit(method='newton')


# In[14]:


model_fitted.summary()


# In[15]:


coef= model_fitted.params
print(coef)


# In[16]:


np.exp(coef)


# In[17]:


#성능평가
train_prob = model_fitted.predict(X_train)
train_prob.head()


# In[18]:


#training 데이터 예측성능
train_prob = model_fitted.predict(X_train)
train_results = pd.concat([train_prob, y_train], axis=1)
train_results.columns = ['Predicted Probability of Class 1', 'Status']
display(train_results)


# In[19]:


#test 데이터 예측성능
test_prob = model_fitted.predict(X_test)
test_results = pd.concat([test_prob, y_test], axis=1)
test_results.columns = ['Predicted Probability of Class 1', 'Status']
display(test_results)


# In[20]:


train_class = train_prob.copy()
test_class = test_prob.copy()

train_class[train_class > 0.5] = 1
train_class[train_class <= 0.5] = 0

test_class[test_class > 0.5] = 1
test_class[test_class <= 0.5] = 0


# In[21]:


# Traiing Accuracy
#accuracy_score(y_train, train_class)
display(pd.DataFrame(confusion_matrix(y_train, train_class), columns=[0,1], index=[0,1]))
print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, train_class)))


# In[22]:


# Test Accuracy
#accuracy_score(y_test, test_class)
display(pd.DataFrame(confusion_matrix(y_test, test_class), columns=[0,1], index=[0,1]))
print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, test_class)))


# In[26]:


train_class_2 = train_prob.copy()
test_class_2 = test_prob.copy()

train_class_2[train_class_2 > 0.3] = 1
train_class_2[train_class_2 <= 0.3] = 0

test_class_2[test_class_2 > 0.3] = 1
test_class_2[test_class_2 <= 0.3] = 0

print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train, train_class_2)))
display(pd.DataFrame(confusion_matrix(y_train, train_class_2), columns=[0,1], index=[0,1]))
print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test, test_class_2)))
display(pd.DataFrame(confusion_matrix(y_test, test_class_2), columns=[0,1], index=[0,1]))


# In[27]:


train_class_3 = train_prob.copy()
test_class_3 = test_prob.copy()

train_class_3[train_class_3 > 0.1] = 1
train_class_3[train_class_3 <= 0.1] = 0

test_class_3[test_class_3 > 0.1] = 1
test_class_3[test_class_3 <= 0.1] = 0

print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train, train_class_3)))
display(pd.DataFrame(confusion_matrix(y_train, train_class_3), columns=[0,1], index=[0,1]))
print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test, test_class_3)))
display(pd.DataFrame(confusion_matrix(y_test, test_class_3), columns=[0,1], index=[0,1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




