#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[32]:


data = pd.read_csv('UniversalBank.csv')
data.head()


# In[35]:


data = data.drop(['ID','ZIP Code'], axis=1)
data.head()


# In[33]:


#데이터크기와 결측치 확인

print(data.shape)
print(data.isnull().sum())


# In[34]:


data.columns


# In[39]:


#수치형 X변수의 히스토그램

numerical_columns = ['Age', 'Experience', 'Family', 'CCAvg',
       'Education', 'Mortgage']
     

fig = plt.figure(figsize = (16, 20))
ax = fig.gca()
data[numerical_columns].hist(ax=ax)
plt.show()


# In[40]:


# Person 상관계수
cols = ['Age', 'Experience', 'Family', 'CCAvg',
       'Education', 'Mortgage']

corr = data[cols].corr(method = 'pearson')
corr


# In[41]:


# heatmap (seaborn)
fig = plt.figure(figsize = (16, 12))
ax = fig.gca()

sns.set(font_scale = 1.5)  # heatmap 안의 font-size 설정
heatmap = sns.heatmap(corr.values, annot = True, fmt='.2f', annot_kws={'size':15},
                      yticklabels = cols, xticklabels = cols, ax=ax, cmap = "RdYlBu")
plt.tight_layout()
plt.show()


# In[44]:


#데이터 전처리
from sklearn.preprocessing import StandardScaler

# 변수 표준화 

scaler = StandardScaler()  # 평균 0, 표준편차 1
scale_columns = ['Age', 'Experience', 'Family', 'CCAvg',
       'Education', 'Mortgage']
data[scale_columns] = scaler.fit_transform(data[scale_columns])


# In[43]:


data.head()


# In[45]:


# 학습데이터 테스트데이터 나누기

from sklearn.model_selection import train_test_split

# split dataset into training & test
X = data[numerical_columns]
y = data['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[46]:


#다중공선성 

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['features'] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif.round(1)


# In[48]:


#화귀 모델링
from sklearn import linear_model

# fit regression model in training set
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# predict in test set
pred_test = lr.predict(X_test)


# In[50]:


# 회귀 계수, 회귀식의 적합도 및 모델 해석
print(lr.coef_)

# 회귀 계수 DataFrame 만들기
coefs = pd.DataFrame(zip(data[numerical_columns].columns, lr.coef_), columns = ['feature', 'coefficients'])
coefs


# In[51]:


# 정렬
coefs_new = coefs.reindex(coefs.coefficients.abs().sort_values(ascending=False).index)
coefs_new


# In[52]:


import statsmodels.api as sm

X_train2 = sm.add_constant(X_train)
model2 = sm.OLS(y_train, X_train2).fit()
model2.summary()


# In[59]:


#모델 예측 및 성능평가
# 예측 결과 시각화 (test set)
df = pd.DataFrame({'actual': y_test, 'prediction': pred_test})
df = df.sort_values(by='actual').reset_index(drop=True)

plt.figure(figsize=(12, 9))
plt.scatter(df.index, df['prediction'], marker='x', color='r')
plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
plt.title("Prediction Result in Test Set", fontsize=20)
plt.legend(['prediction', 'actual'], fontsize=12)
plt.show()


# In[60]:


# RMSE

from sklearn.metrics import mean_squared_error
from math import sqrt

# training set

pred_train = lr.predict(X_train)
print('-'*10,'Training error','-'*10)
print(sqrt(mean_squared_error(y_train, pred_train)))

# test set
print('-'*10,'Test error','-'*10)
print(sqrt(mean_squared_error(y_test, pred_test)))


# In[ ]:





# In[ ]:




