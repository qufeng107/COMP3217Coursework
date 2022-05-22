#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import regularizers
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[2]:


# read dataset

hours = list(map(str, range(24))) 
prices_columns = list(map(str, range(24))) + ['label']
prices = pd.read_csv('./TrainingData.txt', names = prices_columns)


# In[3]:


# 80% as training data, 20% as testing data
train_x = prices.sample(frac=0.8, random_state=7)
train_y = train_x.pop('label')

test_x = prices.drop(train_x.index)
test_y = test_x.pop('label')


# data normalization

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_x)
scaled_test = scaler.fit_transform(test_x)


# In[4]:


# create model

model = tf.keras.Sequential()


# In[5]:


# add layers


model.add(tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=regularizers.l2(0.002), input_shape=(24,)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=regularizers.l2(0.002)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=regularizers.l2(0.003)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(16,activation='relu', kernel_regularizer=regularizers.l2(0.004)))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))


# In[6]:


# compile model

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# In[7]:


# fitting

history = model.fit(scaled_train, train_y, batch_size = 256, epochs = 200, validation_split=0.25)


# In[8]:


# testing data

test_prediction = model.predict(scaled_test)


# In[9]:


# mean_squared_error of testing data

print("MSE: ", metrics.mean_squared_error(test_y, test_prediction))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(test_y, test_prediction)))


# In[10]:


import matplotlib.pyplot as plt

history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.figure(dpi = 200, figsize = (15,5))
plt.plot(epochs,loss_values,'bo', markersize=1,label='Training loss')
plt.plot(epochs,val_loss_values,'b',markersize=0.5,label='Validation loss')
plt.title('Data loss of each epoch')
plt.xlabel('Epochs')
plt.ylabel('Data loss')
plt.legend()


# In[11]:


# read data that need to be calculated

test_prices = pd.read_csv('./TestingData.txt', names = hours)


# In[12]:


# data normalization

scaled_test = scaler.fit_transform(test_prices)

# predict and calculate label using model

test_prediction = (model.predict(scaled_test) > 0.5).astype("int8").flatten()

print(test_prediction)


# In[15]:


# process results and output to 'TestingResults.txt'

test_prediction = pd.concat([test_prices, pd.Series(test_prediction, name='label')], axis=1)
test_prediction.to_csv(os.path.join('./', 'TestingResults.txt'),header=False, index=False)

