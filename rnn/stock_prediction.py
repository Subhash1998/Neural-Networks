#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:35:44 2019

@author: subhash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Importing training set
training_set = pd.read_csv('/home/subhash/neural networks/rnn/trainset.csv')
training_set = training_set.iloc[:,1:2].values

#Feature Scaling
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#Getting inputs and outputs
X_train = training_set[0:1258]
y_train = training_set[1:1259]    

#Reshaping
X_train = np.reshape(X_train,(1258,1,1))

regressor = Sequential()

#input layer
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

#output layer
regressor.add(Dense(units=1))

#compiling rnn
regressor.compile(optimizer='adam',loss='mean_squared_error')

#fitting the model
regressor.fit(x=X_train,y=y_train,batch_size=32,epochs=200)

#making predictions
test_set = pd.read_csv('/home/subhash/neural networks/rnn/testset.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(125,1,1))

predicted_stock_price = regressor.predict(inputs)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

