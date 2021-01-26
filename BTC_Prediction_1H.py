# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:15:43 2021

@author: Mattia
"""

import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM




# Import the dataset and encode the date
df = pd.read_csv('http://www.cryptodatadownload.com/cdd/gemini_BTCUSD_1hr.csv', header=1) 



last_day = df['Date'].head(24)
last_day = last_day[:: -1]

df['Date'] = pd.to_datetime(df['Date']).dt.date




del df['Unix Timestamp']
del df['Symbol']
del df['Open']
del df['High']
del df['Low']
del df['Volume']

df = df[:: -1]
price = df['Close']

# split data
prediction_hours = 24
df_test= price[len(price)-prediction_hours:]
df_train= price[:len(price)-prediction_hours]


training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))




# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 5, epochs = 10)






test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_BTC_price = regressor.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)






# Visualising the results
df = df[:: -1]
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction', fontsize=40)
df_test = df_test.reset_index()
x=df_test.index
labels = last_day
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()