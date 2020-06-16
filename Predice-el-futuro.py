#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('train_csv.csv',index_col=[1],parse_dates=True,squeeze=True)
train.head()


# In[3]:


test = pd.read_csv('test_csv.csv')
test.head()


# In[4]:


train.isnull().sum()


# In[5]:


test.isnull().sum()


# In[6]:


plt.plot(train)


# In[7]:


train['feature'].describe()


# In[8]:


#smoothing of time series - moving average


# In[9]:


train['diff'] = train['feature'] - train['feature'].shift(1)


# In[10]:


plt.plot(train['diff'])


# In[11]:


train.dropna(inplace=True)


# In[12]:


from sklearn.metrics import mean_squared_error


# In[13]:


mse = mean_squared_error(train['feature'],train['diff'])
mse


# In[14]:


np.sqrt(mse)


# In[15]:


from statsmodels.tsa.stattools import adfuller 


# In[16]:


adfuller(train['feature'])


# In[17]:


def adfuller_test(feature):
    label = ['ADF statistics','P value','lags','Number of observations']
    result = adfuller(feature)
    for label,value in zip(label,result):
        print(label+ ':' +str(value))
    if result[1] <= 0.05:
        print('accept H1')
    else:
        print('accept H0')


# In[18]:


adfuller_test(train['diff'])


# In[19]:


# P - value is less than 0.05 and rejected null hypothesis
# 


# In[20]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# In[21]:


#p=2, q=1 , d=0-2

ax1=plot_acf(train['diff'])
ax2=plot_pacf(train['diff'])


# In[22]:


train_pred = train[0:63]
test_pred = train[63:]


# In[23]:


train_pred.drop(['feature','id'],axis=1,inplace=True)
test_pred.drop(['feature','id'],axis=1,inplace=True)


# In[24]:


from statsmodels.tsa.arima_model import ARIMA


# In[25]:


#p=2,q=1,d=0-2
model = ARIMA(train_pred,order=(2,0,1))


# In[26]:


model_fit = model.fit()


# In[27]:


model_fit.aic


# In[28]:


model_fit.summary()


# In[29]:


model_forecast = model_fit.forecast(steps=16)[0]


# In[30]:


np.sqrt(mean_squared_error(test_pred,model_forecast))


# In[31]:


test_pred = pd.DataFrame(model_forecast, columns= ['feature'])


# In[32]:


new_test = pd.concat([test, test_pred], axis=1, join='inner')

