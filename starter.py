#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[27]:


import pickle
import pandas as pd
import numpy as np
import datetime


# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[4]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[30]:


df=read_data('./data/fhv_tripdata_2021-02.parquet')


# In[31]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[32]:


print(y_pred.mean())


# In[33]:


df


# In[21]:


year = pd.DatetimeIndex(df['pickup_datetime']).year
month = pd.DatetimeIndex(df['pickup_datetime']).month

year = year.astype(str)
month = month.astype(str)

# df['ride_id'] = f'2021/02_' + df.index.astype('str')


# In[52]:


df["ride_id"]=df.apply(lambda x: f'{x.pickup_datetime.year}/{x.pickup_datetime.month}_'+str(x.name),axis=1)


# In[54]:


df["pred"]=y_pred


# In[55]:


df


# In[60]:


df[["ride_id","pred"]].to_parquet(
    'output_file.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[63]:


get_ipython().system(' jupyter nbconvert --to python starter.ipynb')


# In[64]:


get_ipython().system(' pip install pipenv')


# Hashes

# "scikit-learn": {
#             "hashes": [
#                 "sha256:0403ad13f283e27d43b0ad875f187ec7f5d964903d92d1ed06c51439560ecea0",
#                 "sha256:102f51797cd8944bf44a038d106848ddf2804f2c1edf7aea45fba81a4fdc4d80",
#                 "sha256:22145b60fef02e597a8e7f061ebc7c51739215f11ce7fcd2ca9af22c31aa9f86",
#                 "sha256:33cf061ed0b79d647a3e4c3f6c52c412172836718a7cd4d11c1318d083300133",

# In[ ]:




