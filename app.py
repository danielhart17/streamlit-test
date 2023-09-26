import uvicorn
import pickle
import numpy as np
from fastapi import FastAPI
from PriceFeatures import PriceFeatures

fastAPIapp = FastAPI()

pickle_in = open("model.pkl", "rb")
priceModel = pickle.load(pickle_in)

@fastAPIapp.get('/')
def index():
    return {'message': 'Hello, World'}

@fastAPIapp.post('/predict')
def predict_price(data: PriceFeatures):
    data = data.model_dump()
    open = data['open']
    high = data['high']
    low = data['low']
    rsi = data['rsi']
    zlema = data['zlema']
    stochrsi = data['stochrsi']
    percent_b = data['percent_b']
    qstick = data['qstick']
    prediction = priceModel.predict([[open, high, low, rsi, zlema, stochrsi, percent_b, qstick]])
    return {'prediction': float(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(fastAPIapp, host='0.0.0.0', port=10000)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import datetime
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import yfinance as yf

# st.title('Streamlit Practice')

# train_start = '2020-01-01'
# train_end = '2023-01-01'
# test_start = '2023-01-01'
# test_end = '2023-09-01'

# # Load the data
# # eurusd_df = pd.read_csv('EURUSD=X.csv')
# @st.cache_data
# def load_data(_trainStart, _trainEnd, _testStart, _testEnd):
#   train_df = yf.download('EURUSD=X', _trainStart, _trainEnd, interval='1d')
#   test_df = yf.download('EURUSD=X', _testStart, _testEnd, interval='1d')
#   return train_df, test_df

# eurusd_df, eurusd_df_new = load_data(train_start, train_end, test_start, test_end)

# st.subheader('Original Data')
# st.write(eurusd_df.head())

# # eurusd_df.reset_index(inplace=True)
# # st.write(eurusd_df.head())

# # # function to convert date string to datetime object
# # def str_to_datetime(s):
# #   split = s.split('-') #determine where to split the date string 
# #   year, month, day = int(split[0]), int(split[1]), int(split[2]) # convert the different splits of date string into integers and set them to year, month, and day variables respectively
# #   return datetime.datetime(year=year, month=month, day=day) #return the year month and day splits as a datetime object 

# # # convert the date column to datetime objects
# # eurusd_df['Date'] = eurusd_df['Date'].apply(str_to_datetime)

# # set the index of the dataframe to the date column
# # eurusd_df.index = eurusd_df.pop('Date')

# # separate the close column from the rest of the data
# eurusd_closes = eurusd_df['Close']

# # drop the adj close column and the volume column from the dataframe
# eurusd_data = eurusd_df.drop(['Adj Close', 'Volume', 'Close'], axis=1)

# st.subheader('Closes')
# st.write(eurusd_closes.head())
# st.subheader('Data')
# st.write(eurusd_data.head())

# n_steps = 5 #steps to step through dataframe 
# features = [] #create an empty list for features
# labels = [] #create an empty list for labels 
# for i in range(len(eurusd_closes)-n_steps):
#   features.append(eurusd_data.iloc[i:i+n_steps].values)
#   labels.append(eurusd_closes.iloc[i+n_steps])
# features = np.array(features)
# labels = np.array(labels)

# st.text('Features shape: {features.shape}')
# st.write(features)

# st.text('Labels shape: {labels.shape}')
# st.write(labels)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=7)


# # Create Model
# tf.random.set_seed(7)

# multi_variate_price_model_ttc = tf.keras.Sequential([
#     tf.keras.layers.LSTM(500, return_sequences=True),
#     tf.keras.layers.LSTM(200, return_sequences=True), 
#     tf.keras.layers.LSTM(100, return_sequences=True), 
#     tf.keras.layers.LSTM(50, return_sequences=True), 
#     tf.keras.layers.LSTM(25, return_sequences=True), 
#     tf.keras.layers.Flatten(), 
#     tf.keras.layers.Dense(500, activation = 'relu'),
#     tf.keras.layers.Dense(200, activation = 'relu'),
#     tf.keras.layers.Dense(100, activation = 'relu'),
#     tf.keras.layers.Dense(50, activation = 'relu'),
#     tf.keras.layers.Dense(1)
# ])


# # configure the model
# multi_variate_price_model_ttc.compile(loss=tf.keras.losses.mae, 
#                                   optimizer=tf.keras.optimizers.Adam(0.00011), 
#                                   metrics = ["mae", "mse"])
# # train the model
# history = multi_variate_price_model_ttc.fit(X_train, y_train, epochs=20)

# st.subheader('Model Loss')
# st.line_chart(history.history['loss'])

# # save the model
# multi_variate_price_model_ttc.save('multi_variate_price_model_ttc.h5')

# # evaluate the model
# evaluation = multi_variate_price_model_ttc.evaluate(X_test, y_test, return_dict=True)
# st.line_chart(evaluation)

# # make predictions
# predictions = multi_variate_price_model_ttc.predict(X_test)
# st.line_chart(predictions)

# st.subheader('Test Data')
# st.write(eurusd_df_new.head())

