import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fredapi import Fred 
import keras_tuner as kt 
from finta import TA 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import datetime 
from datetime import datetime, timedelta
#from tensorflow.python.ops.numpy_ops import np_config
# import v20 
import requests
import seaborn as sns 
import streamlit as st 
import json 
#np_config.enable_numpy_behavior()
import time 
import os 
# import schedule 
import pytz
from datetime import date
from dateutil.relativedelta import relativedelta
import pickle


start_time = datetime.now(pytz.timezone('US/Eastern')).replace(hour=4, minute=58, second=30, microsecond=0).time()
end_time = datetime.now(pytz.timezone('US/Eastern')).replace(hour=16, minute=0, second=0, microsecond=0).time()
#while True:
start_date = datetime(2023, 8, 1)
end_date = datetime(2023, 8, 30)
ticker = ["GBPUSD=X"]
#fred api key and client 

fred_key = '9844e0e13499c4426a0cae456bbfc950'
fred = Fred(api_key=fred_key) #used to pull data from fred website 
#gets fred data that comes in as a series and turns it into a Dataframe to make it easier to merge
def fred_dataframe_creator(series, column_name):
    df = fred.get_series(series)
    df = pd.DataFrame(df)
    df = df.rename(columns={0:column_name})

    return df
#same fred dataframe creator but this one allows you to specify start and end date for series 

def fred_dataframe_creator_2(series, column_name, start, end):
    start = start 
    end = end 
    df = fred.get_series(series, observation_start=start, observation_end=end)
    df = pd.DataFrame(df)
    df = df.rename(columns={0:column_name})

    return df

#function to take closes away from price data and make features and labels
def features_and_labels(prices):
    features, labels = prices.drop(['Adj Close', 'Close'], axis=1), prices['Adj Close']
    print(features.dtypes)
    return features, labels

#function to get prices from yahoo finance and add indicator data from finta if any are passed 
def get_prices(ticker, indicators, start, end, interval, period, forex):
    # import data from yahoo finance
    price = yf.download(tickers=(ticker), start=(start), end=(end), interval=(interval))    
    
    # remove volume column
    if forex == True:
        price = price.drop(columns=['Volume'])
  
    # add indicators
    for indicator in indicators:
        if indicator == 'RSI':
            price['RSI'] = TA.RSI(price, period)
        elif indicator == 'ZLEMA':
            price['ZLEMA'] = TA.ZLEMA(price, period)
        elif indicator == 'MFI':
            price['MFI'] = TA.MFI(price, period)
        elif indicator == 'BBWIDTH':
            price['BBWIDTH'] = TA.BBWIDTH(price, period)
        elif indicator == 'CCI':
            price['CCI'] = TA.CCI(price, period)
        elif indicator == 'OBV':
            price['OBV'] = TA.OBV(price)
        elif indicator == 'ROC':
            price['ROC'] = TA.ROC(price, period)
        elif indicator == 'AO':
            price['AO'] = TA.AO(price, slow_period=34, fast_period=5)
        elif indicator == 'CFI':
            price['CFI'] = TA.CFI(price)
        elif indicator == 'CHAIKIN':
            price['CHAIKIN'] = TA.CHAIKIN(price)
        elif indicator == 'COPPOCK':
            price['COPPOCK'] = TA.COPP(price)
        elif indicator == 'ER':
            price['ER'] = TA.ER(price, period)
        elif indicator == 'EVSTC':
            price['EVSTC'] = TA.EVSTC(price)
        elif indicator == 'IFT_RSI':
            price['IFT_RSI'] = TA.IFT_RSI(price)
        elif indicator == 'MI':
            price['MI'] = TA.MI(price, period)
        elif indicator == 'PERCENT_B':
            price['PERCENT_B'] = TA.PERCENT_B(price, period)
        elif indicator == 'PZO':
            price['PZO'] = TA.PZO(price, period)
        elif indicator == 'QSTICK':
            price['QSTICK']  = TA.QSTICK(price, 21)
        elif indicator == 'SAR':
            price['SAR'] = TA.SAR(price)
        elif indicator == 'SMM':
            price['SMM'] = TA.SMM(price, period)
        elif indicator == 'STC':
            price['STC'] = TA.STC(price)
        elif indicator == 'STOCH':
            price['STOCH'] = TA.STOCH(price, period)
        elif indicator == 'STOCHRSI':
            price['STOCHRSI'] = TA.STOCHRSI(price, period, period)
        elif indicator == 'SWI':
            price['SWI'] = TA.SWI(price, period)
        elif indicator == 'VZO':
            price['VZO'] = TA.VZO(price, period)


        # Add more elif statements for other indicators
        #Look at using a dictionary with indicator value as key and TA.Key as the value pair
        
    price = price.dropna()
    return price

# data processor splits as of now because we need to scale outside of a function if we want to inverse scale prices back 
def data_processor(scaler, splitter, splits, range, data, labels):
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    if splitter == TimeSeriesSplit:
        splitter = TimeSeriesSplit(n_splits=int(splits))
    if scaler == MinMaxScaler:
        scaler = MinMaxScaler(feature_range=range)
    for train, test in splitter.split(data):
        X_train, X_test, y_train, y_test = data.iloc[train, :], data.iloc[test, :], labels.iloc[train], labels.iloc[test]
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    X_train, X_test, y_train, y_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
    #X_train, X_test, y_train, y_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test), scaler.fit_transform(y_train.reshape(-1, 1)), scaler.fit_transform(y_test.reshape(-1, 1))
    #print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

# function to reshape data when number of columns of dataframe change
def lstm_reshaper(X_train, X_test, y_train, y_test,df):
    num_cols = int(len(df.columns))
    train_size = X_train.size
    test_size = X_test.size
    first_train_num = int(train_size/num_cols)
    first_test_num = int(test_size/num_cols)

    X_train, X_test, y_train, y_test = tf.reshape(X_train, shape=(first_train_num, num_cols, 1)), tf.reshape(X_test, shape=(first_test_num, num_cols, 1)), tf.reshape(y_train, shape=(first_train_num)), tf.reshape(y_test, shape=(first_test_num))
    return X_train, X_test, y_train, y_test

# function to get the most recent set of features to pass to our model to make new predictions each day
def get_most_recent_value(X_test):
    X_test = X_test[-1:]
    return X_test 

def inverse_scaler(x, scaler):
    x = scaler.inverse_transform(x)
    return x

# model itself 
def odc_intraday_model_official_v2():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True,activation='tanh'), 
        tf.keras.layers.LSTM(100, return_sequences=True,activation='tanh'), 
        tf.keras.layers.LSTM(100, return_sequences=True,activation='tanh'), 
        tf.keras.layers.LSTM(100, return_sequences=True,activation='tanh'), 
        tf.keras.layers.LSTM(100, return_sequences=False, activation='tanh'), 
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
    return model 


prices = get_prices(ticker=ticker, indicators=['RSI', 'ZLEMA', 'STOCHRSI', 'PERCENT_B', 'QSTICK'], start=start_date, end=end_date, interval='15m', period=21, forex=True)
# print(prices)
features, labels = features_and_labels(prices=prices)
scaler = MinMaxScaler(feature_range=(0, 1))
callbacks = tf.keras.callbacks.EarlyStopping(monitor="mae", min_delta=0, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=True)
X_train, X_test, y_train, y_test = data_processor(scaler=MinMaxScaler, splitter=TimeSeriesSplit, splits=5, range=(0, 1), data=features, labels=labels)
X_train, X_test, y_train, y_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test), scaler.fit_transform(y_train.reshape(-1, 1)), scaler.fit_transform(y_test.reshape(-1, 1))
X_train, X_test, y_train, y_test = lstm_reshaper(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, df=features)
print(X_train.shape)
model = odc_intraday_model_official_v2()

# now = datetime.now(pytz.timezone('US/Eastern'))
# if (now.weekday() >= 4 and now.weekday() <= 6) and (now.time() >= start_time and now.time() <= end_time):
history = model.fit(X_train, y_train, epochs=200, callbacks=callbacks)

# pickle.dump(model, open('model.pkl', 'wb'))
preds = model.predict(X_test)


preds = scaler.inverse_transform(preds)
#print(preds)
y_test_inv = tf.reshape(y_test, shape=(-1, 1))
y_test_inv = scaler.inverse_transform(y_test_inv)
results = pd.DataFrame({'y':y_test_inv.flatten(), 'preds':preds.flatten(), 'diff':((y_test_inv.flatten()) - (preds.flatten()))})
# print(results)
plt.figure(figsize=(10, 7))
#plot the training preds vs actuals to see how well our model learned the training patterns
def plot_preds_v_actuals(df):
    plt.plot(df['y'], c="b", label='Actual')
    plt.plot(df['preds'], c="red", label="Predicted" )
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

#plot the loss function 
def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['mae'], label='MAE')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the chart
    plt.show()
# today = datetime.now()
# now = datetime.now()
# current = datetime.now()

# date = current - timedelta(days=60)

#     # Calculate the ending date 60 days in the past


#     # Calculate the starting date 60 days before the end date, but include the current time
# if (now.weekday() >= 1 and now.weekday() <= 4) and (now.time() >= start_time and now.time() <= end_time):
#     new_end = datetime.now()

#     new_start = new_end - timedelta(days=1)
#     new_prices = get_prices(ticker=ticker, indicators=['RSI', 'ZLEMA', 'STOCHRSI', 'PERCENT_B', 'QSTICK'], start=new_start, end=new_end, interval='15m', period=21, forex=False)
#     new_features, new_labels = features_and_labels(prices=new_prices)
#     new_X_train, new_X_test, new_y_train, new_y_test = data_processor(scaler=MinMaxScaler, splitter=TimeSeriesSplit, splits=5, range=(0, 1), data=new_features, labels=new_labels)
#     new_X_train, new_X_test, new_y_train, new_y_test = scaler.fit_transform(new_X_train), scaler.fit_transform(new_X_test), scaler.fit_transform(new_y_train.reshape(-1, 1)), scaler.fit_transform(new_y_test.reshape(-1, 1))
#     new_X_train, new_X_test, new_y_train, new_y_test = lstm_reshaper(X_train=new_X_train, X_test=new_X_test, y_train=new_y_train, y_test=new_y_test, df=new_features)
#     predictive_value = new_X_test[-1:]

#     def get_prediction(value=predictive_value):
#         new_preds = model.predict(value)
#         return new_preds 

#     new_preds = get_prediction(value=predictive_value)
#     new_preds = scaler.inverse_transform(new_preds)
#     print(new_features)
#     print(new_preds)
#     current_close = new_prices['Adj Close']
#     current_close = current_close[-1]
#     current_pred = new_preds[0][0]
#     API = "api-fxtrade.oanda.com"
#     STREAM_API = "stream-fxtrade.oanda.com"
#     ACCESS_TOKEN = "01444768cd6656f2f56f54ccf41c3cc1-9c7f11165bd65ef3dd6cd0bbf7ab8c8a"
#     ACCOUNT_ID = "001-001-2733676-001"  
#     PRICING_PATH = f"/v3/accounts/{ACCOUNT_ID}/pricing"
#     pending_order_path = f"/v3/accounts/{ACCOUNT_ID}/pendingOrders"
#     order_path = f"/v3/accounts/{ACCOUNT_ID}/Orders"
#     query = {"instruments":"GBP_USD"} 
#     headers = {"Authorization": "Bearer "+ ACCESS_TOKEN, "Content-Type": "application/json"}
#     response = requests.get("https://"+API+PRICING_PATH, headers=headers, params=query)
#     order_response = requests.get("https://"+API+pending_order_path, headers=headers, params=query)
#     open_order_response = requests.get("https://"+API+order_path, headers=headers, params=query)
#     url = "https://" + API + "/v3/accounts/" + ACCOUNT_ID + "/orders"
#     headers = {"Authorization" : "Bearer "  + ACCESS_TOKEN}
#     instrument = "GBP_USD"

#     #create market buy orders
#     test_buy_mkt = {
#     "order": {
#         "units": "37",
#         "trailingStopLossOnFill": {
#         "distance":"0.0010"
#         },
#         "instrument": "GBP_USD",
#         "timeInForce": "FOK",
#         "type": "MARKET",
#         "positionFill": "DEFAULT"
#     }
#     }
#     #create market sell orders
#     test_sell_mkt = {
#     "order": {
#         "units": "-37",
#         "trailingStopLossOnFill": {
#         "distance":"0.0010"
#         },
#         "instrument": "GBP_USD",
#         "timeInForce": "FOK",
#         "type": "MARKET",
#         "positionFill": "DEFAULT"
#     }
#     }

#     # message = "Order Entered go update take profit"
#     # title = "Notification"
#     test_response = order_response.json() 
#     open_orders = test_response['orders']
#     difference_in_preds_too_small_bullish = False 
#     difference_in_preds_too_small_bearish = False 
#     # # account_sid = 'your_account_sid'
#     # # auth_token = 'your_auth_token'
#     # # client = Client(account_sid, auth_token)
#     # # my_phone_number = '+1234567890'
#     # # twilio_phone_number = '+0987654321'
#     # def get_prediction(value=predictive_value):
#     #     new_preds = model.predict(value)
#     #     return new_preds 
#     # test_preds = get_prediction(value=predictive_value)
#     #print(current_prices)

#     #print(test_preds)



#     def check_and_create_order(pred, close, orders, order_response):
#         difference_in_preds_too_small_bullish = False 
#         difference_in_preds_too_small_bearish = False 
#         # pred = float(pred)
#         # close = float(close)
#         if pred - close <= 0.0001:
#             difference_in_preds_too_small_bullish = True 
#         if pred < close:
#             if close - pred <= 0.0001:
#                 difference_in_preds_too_small_bearish = True 
#         #check if orders are open, if there arent any pending or open orders check if we are getting the right response from the oanda api call, if we are check for entry


#         if orders:
#             order_title = "No Order"
#             order_message = "Orders already entered"
#             print("Order Pending or Open") 
#             print(open_orders)
#             os.system(f"terminal-notifier -title '{order_title}' -message '{order_message}' - orders'{order_response.json()}'")
#         elif order_response.status_code == 400:
#             print("Error", order_response.text)
#         elif order_response.status_code == 404:
#             print("Error", order_response.text)
#         elif order_response.status_code == 405:
#             print("Error", order_response.text)
#         else:


#             if pred > close:
#                 diff_title = "Difference too small"
#                 diff_message = "No order open but predicted price change is too small"
#                 if difference_in_preds_too_small_bullish:
#                     print("Too small difference")
#                 else:
#                     requests.post(url, headers=headers, json=test_buy_mkt)
#                     print("Buy")

#             elif pred < close:
#                     diff_title = "Difference too small"
#                     diff_message = "No order open but predicted price change is too small"
#                     if difference_in_preds_too_small_bearish:
#                         print("Too small difference")


#                     else:
#                         requests.post(url, headers=headers, json=test_sell_mkt)
#                         print("Sell")
#     check_and_create_order(pred=current_pred, close=current_close, orders=open_orders, order_response=order_response)





# time.sleep(900)