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
training_data_len = math.ceil(len(values) * 0.5)

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

# Crie uma nova entrada de teste para o dia seguinte
last_60_days = scaled_data[-60:]
x_test_next_day = np.reshape(last_60_days, (1, 60, 1))
prediction_next_day = model.predict(x_test_next_day)
prediction_next_day = scaler.inverse_transform(prediction_next_day)

# Adicione a previsão do próximo dia ao DataFrame
next_day = pd.DataFrame({'Close': [prediction_next_day[0, 0]]}, index=[stock_data.index[-1] + timedelta(days=1)])
stock_data = pd.concat([stock_data, next_day])

# Faça previsões para o conjunto de teste original
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calcule o RMSE original e para o próximo dia
rmse_original = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse_next_day = np.sqrt(np.mean(prediction_next_day - y_test[-1]) ** 2)

data = stock_data.filter(['Close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = np.concatenate([predictions, prediction_next_day])

fig = go.Figure()

fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=validation.index, y=validation['Close'], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=validation.index, y=validation['Predictions'], mode='lines', name='Predictions'))

fig.update_layout(
    title='Stock Price Prediction with LSTM',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Close Price USD ($)'),
    legend=dict(x=0, y=1, traceorder='normal'),
    annotations=[
        dict(x=train.index[-1], y=train['Close'].iloc[-1], xref="x", yref="y", text="Last Train Data", showarrow=True, arrowhead=5, ax=0, ay=-40),
        dict(x=validation.index[0], y=validation['Close'].iloc[0], xref="x", yref="y", text="First Validation Data", showarrow=True, arrowhead=5, ax=0, ay=-40),
        dict(x=validation.index[-1], y=validation['Predictions'].iloc[-1], xref="x", yref="y", text="Last Prediction", showarrow=True, arrowhead=5, ax=0, ay=-40),
    ]
)

print(f'RMSE for the original test set: {rmse_original}')
print(f'Prediction for the next day: {prediction_next_day[0, 0]} (RMSE: {rmse_next_day})')

fig.show()
