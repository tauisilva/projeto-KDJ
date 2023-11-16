# Importa bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Define o caminho do arquivo CSV
cvs = "./Colab/AAPL.csv"

# Lê o arquivo CSV e cria um DataFrame Pandas
df = pd.read_csv(cvs)

df.head()

# Verifica se há valores nulos em qualquer linha
df[df.isna().any(axis=1)]

# Remove linhas com valores nulos
df = df.dropna()

# Obtém o tamanho do DataFrame após a remoção dos valores nulos
len(df)

# Extrai a série temporal de preços de fechamento
prices = df["Close"]

# Plota o histórico de preços
plt.figure(figsize=(18, 9))
plt.plot(df["Close"])
plt.xticks(range(0, df.shape[0], 100), df["Date"].loc[::100], rotation=45)
plt.xlabel("Datas", fontsize=18)
plt.ylabel("Preço Fechamento", fontsize=18)
plt.title("Histórico de Preço PETR4", fontsize=30)
plt.show()

# Define o tamanho do passo temporal para a sequência
days_time_step = 15

# Divide os dados em conjunto de treinamento e teste
training_size = int(len(prices) * 0.95)
test_size = len(prices) - training_size
train_data, input_data = np.array(prices[0:training_size]), np.array(
    prices[training_size - days_time_step :]
)
test_data = np.array(prices[training_size:])

# Plota os conjuntos de treinamento e teste
plt.figure(figsize=(18, 9))
plt.plot(df["Close"].loc[0 : train_data.shape[0]], color="blue", label="treino")
plt.plot(df["Close"].loc[train_data.shape[0] :], color="red", label="teste")
plt.xticks(range(0, df.shape[0], 100), df["Date"].loc[::100], rotation=45)
plt.xlabel("Datas", fontsize=18)
plt.ylabel("Preço Médio", fontsize=18)
plt.title("Histórico de Preço PETR4", fontsize=30)
plt.legend()
plt.show()

# Inicializa o normalizador para escalonar os dados entre 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Escalona os dados de treinamento, teste e validação
train_data_norm = scaler.fit_transform(np.array(train_data).reshape(-1, 1))
test_data_norm = scaler.transform(np.array(input_data).reshape(-1, 1))
val_data_norm = scaler.transform(np.array(test_data).reshape(-1, 1))

# Exibe os dados normalizados de treinamento
train_data_norm

# Inicializa listas vazias para armazenar os dados de treinamento, teste e validação
X_train, y_train = [], []
X_test = []
X_val, y_val = [], []

# Cria sequências para treinamento, teste e validação
for i in range(days_time_step, len(train_data)):
    X_train.append(train_data_norm[i - days_time_step : i])
    y_train.append(train_data_norm[i])

for i in range(days_time_step, days_time_step + len(test_data)):
    X_test.append(test_data_norm[i - days_time_step : i])

for i in range(days_time_step, len(test_data)):
    X_val.append(val_data_norm[i - days_time_step : i])
    y_val.append(val_data_norm[i])

# Converte as listas para matrizes numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Exibe a forma da matriz de teste
X_test.shape

# Obtém a versão do TensorFlow
tf.__version__

# Cria o modelo de rede neural sequencial
model = Sequential()
model.add(SimpleRNN(100, return_sequences=False, input_shape=(days_time_step, 1)))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")

# Exibe a arquitetura do modelo
model.summary()

# Treina o modelo
h = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32
)

# Plota as curvas de perda durante o treinamento
plt.plot(h.history["loss"], label="loss")
plt.plot(h.history["val_loss"], label="val_loss")
plt.legend()

# Faz a previsão dos valores usando o modelo treinado
predict = model.predict(X_test)

# Inverte a escala para obter os valores reais
predict = scaler.inverse_transform(predict)
real = test_data

# Exibe a forma da matriz de previsões
predict.shape

# Plota os resultados da previsão em comparação com os valores reais
plt.figure(figsize=(18, 9))
plt.plot(real, color="green", label="real")
plt.plot(predict, color="red", label="previsão")
plt.xticks(range(0, len(real), 50), df["Date"].iloc[-len(real) :: 50], rotation=45)
plt.xlabel("Datas", fontsize=18)
plt.ylabel("Preço Médio", fontsize=18)
plt.title("Projeção de Preço PETR4", fontsize=30)
plt.legend()
plt.show()

# Calcula o erro médio quadrático entre os valores reais e previstos
from sklearn.metrics import mean_squared_error

mean_squared_error(real, predict)
