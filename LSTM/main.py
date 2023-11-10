import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
from datetime import datetime, timedelta

end_date = datetime.today().date()
start_date = end_date - timedelta(days=365 * 10)

stock_data = yf.download('AAPL', start=start_date, end=end_date)

close_prices = stock_data['Close']
values = close_prices.values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values.reshape(-1, 1))

train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=3)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse

data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions

# Plotly visualization
fig = go.Figure()

fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=validation.index, y=validation['Close'], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=validation.index, y=validation['Predictions'], mode='lines', name='Predictions'))

fig.update_layout(
    title='Model',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Close Price USD ($)'),
    legend=dict(x=0, y=1, traceorder='normal')
)

fig.show()
