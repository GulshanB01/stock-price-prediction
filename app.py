import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import streamlit as st


st.title('🔮 FutureSight: Stock Price Prediction Dashboard')
st.subheader('Interactive Stock Market Forecasting Using Deep Learning (LSTM) and Streamlit')

start = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
end = st.date_input('End Date', pd.to_datetime('2019-12-31'))

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input,start,end,threads=False,auto_adjust=True)

#describe data
st.subheader('Data from 2010-2024')
st.write(df.describe())

ma_window = st.slider('Select Moving Average Window', min_value=10, max_value=200, value=100, step=10)
ma = df.Close.rolling(ma_window).mean()

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close Price')
plt.xlabel('Time')         # X-axis label
plt.ylabel('Price')         # Y-axis label
plt.legend()
st.pyplot(fig1)

st.subheader(f'Closing Price with {ma_window}-Day Moving Average')
fig2 = plt.figure(figsize=(12,6))
plt.plot(ma,'red',label=f'{ma_window}-Day MA')
plt.xlabel('Time')         # X-axis label
plt.ylabel('Price') 
plt.plot(df['Close'],'blue',label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig3=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='100 Day MA')
plt.plot(ma200,'g',label='200 Day MA')
plt.xlabel('Time')         # X-axis label
plt.ylabel('Price')
plt.plot(df['Close'],'b',label='Close Price')
plt.legend()
st.pyplot(fig3)

# Splitting the data into training and testing sets
#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])#interested in close col and till 70% of close value
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#loading the model
model = load_model('keras_model.keras')
# Testing Part
past_100days = data_training.tail(100)
final_df = pd.concat([past_100days,data_testing],ignore_index=True)
input_data = scaler.fit_transform(final_df) 

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)
scaler.scale_
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# Final Graph

st.subheader('Predictions vs Original')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test,'blue',label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# User input for future prediction days
st.subheader('🔮 Predict Future Stock Prices')
future_days = st.number_input('How many future days to predict?', min_value=1, max_value=100, value=30, step=1)

# Predict future prices only if user clicks button
if st.button('Predict Future'):
    
    # Prepare last 100 days data as input
    last_100_days = data_testing.tail(100).values
    last_100_days_scaled = scaler.transform(last_100_days)
    
    X_input = list(last_100_days_scaled)
    future_predictions = []
    
    for i in range(future_days):
        x_input_arr = np.array(X_input[-100:]).reshape(1, 100, 1)
        pred = model.predict(x_input_arr, verbose=0)
        future_predictions.append(pred[0][0])
        X_input.append([pred[0][0]])
    
    # Rescale predictions back to original price scale
    future_predictions = np.array(future_predictions).reshape(-1,1)
    future_predictions = scaler.inverse_transform(future_predictions).flatten()
    
    # Prepare x-axis for future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days + 1, freq='B')[1:]
    
    # Plot Future Predictions
    st.subheader(f'Predicted Stock Price for Next {future_days} Days')
    fig_future, ax_future = plt.subplots(figsize=(12,6))
    ax_future.plot(df['Close'], label='Historical Price')
    ax_future.plot(future_dates, future_predictions, color='red', label='Future Predicted Price')
    ax_future.set_xlabel('Time')
    ax_future.set_ylabel('Price')
    ax_future.legend()
    st.pyplot(fig_future)
