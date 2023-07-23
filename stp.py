#installing required libraries

# !pip install pandas_datareader
# !pip install yfinance
# !pip install yahoo_fin
#!pip install keras
# !pip install tensorflow
# !pip install streamlit-jupyter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()




#importing the data set from yahoo finance which gives us the data from last day to any history 
#importing required libraries


start="2013-06-21"

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Code','AAPL')

df = pdr.get_data_yahoo(user_input, start="2013-06-21")

#Describing the data

st.subheader('Data From 2013')
st.write(df.describe())


#Visualizations

st.subheader('Closing Price VS Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price VS Time Chart With 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price VS Time Chart With 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#Splitting data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#load model

model = load_model('keras_model.h5')

#Predictions/Testing

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test,y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



#final Graph

st.subheader('Predictions VS Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

    
    
# df = df.reset_index()
# df.head()


# df = df.drop(['Date','Adj Close'],axis = 1)
# df

# plt.plot(df.Close)


# ma100 = df.Close.rolling(100).mean()
# ma100


# plt.figure(figsize = (12,6))
# plt.plot(df.Close)
# plt.plot(ma100,'r')


# ma200 = df.Close.rolling(200).mean()
# ma200

# plt.figure(figsize = (12,6))
# plt.plot(df.Close)
# plt.plot(ma100,'r')
# plt.plot(ma200,'g')

# df.shape

# data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# print(data_training.shape)
# print(data_testing.shape)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# data_training_array = scaler.fit_transform(data_training)
# data_training_array

# data_training_array.shape

# x_train = []
# y_train = []

# for i in range(100,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])

# x_train , y_train = np.array(x_train) , np.array(y_train)


# #ML Model

# from keras.layers import Dropout,Dense,LSTM
# from keras.models import Sequential 


# model = Sequential()
# model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
#           input_shape = (x_train.shape[1], 1)))
# model.add(Dropout(0.2))



# model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.3))



# model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.4))



# model.add(LSTM(units = 120, activation = 'relu'))
# model.add(Dropout(0.5))




# model.add(Dense(units = 1))


# model.summary()

# model.compile(optimizer='adam',loss = 'mean_squared_error')
# model.fit(x_train,y_train,epochs=10)

# model.save('keras_model.h5')

# data_testing.head()


# past_100_days = data_training.tail(100)

# final_df = past_100_days.append(data_testing,ignore_index=True)

# final_df.head()

# input_data = scaler.fit_transform(final_df)
# input_data


# input_data.shape

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100: i])
#     y_test.append(input_data[i, 0])


# x_test,y_test = np.array(x_test), np.array(y_test)
# print(x_test.shape)
# print(y_test.shape)


# # Making predictions


# y_predicted = model.predict(x_test)

# y_predicted.shape

# y_test

# y_predicted

# scaler.scale_

# scale_factor = 1/0.00210682
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor


# plt.figure(figsize=(12,6))
# plt.plot(y_test,'b', label = 'Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()