# Resolvendo problemas da lib Ta-Lib
# Fonte: https://stackoverflow.com/a/49660479
# url = 'https://anaconda.org/conda-forge/libta-lib/0.4.0/download/linux-64/libta-lib-0.4.0-h166bdaf_1.tar.bz2'
# !curl -L $url | tar xj -C /usr/lib/x86_64-linux-gnu/ lib --strip-components=1
# url = 'https://anaconda.org/conda-forge/ta-lib/0.4.19/download/linux-64/ta-lib-0.4.19-py310hde88566_4.tar.bz2'
# !curl -L $url | tar xj -C /usr/local/lib/python3.10/dist-packages/ lib/python3.10/site-packages/talib --strip-components=3


# Imports
import talib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import metrics


# Gerando intervalos de tempo automáticamente
data_atual = datetime.today().date()
data_atual_10_anos_atras = data_atual - timedelta(days=365 * 10)

codigo_acao = "googl" # A.K.A "Ticker"
dados_acao_dos_ultimos_10_anos = yf.download(codigo_acao, start=data_atual_10_anos_atras, end=data_atual)


# Exibindo gráfico com os preços de fechamento
dados_acao_dos_ultimos_10_anos['Adj Close'].plot()
plt.ylabel("Preços de Fechamento Corrigidos")
plt.show()

print("\n")

# Calculando a variação percentual entre os dias consecutivos
# e criando um histograma com 50 intervalos para visualizar a
# distribuição das mudanças percentuais.
# O eixo x do histograma representa as mudanças percentuais,
# e o eixo y representa a frequência de ocorrência para cada intervalo.
dados_acao_dos_ultimos_10_anos['Adj Close'].pct_change().plot.hist(bins=50)
plt.xlabel("Variação Percentual do Fechamento Ajustado em 1 Dia")

# O gráfico gerado ilustra como as mudanças percentuais nos preços ajustados de
# uma ação estão distribuídas. Isso pode ser útil para entender a volatilidade da
#  ação ao longo do tempo. A escolha de 50 intervalos nos da mais detalhes sobre a
# distribuição das mudanças percentuais.
plt.show()

print("\n")

# Calculando indicadores de análise técnica para adicionar à lista de dados de treinamento do modelo
# mma = média móvel aritmética
# Médias móveis utilizadas: 21, 50, 80 e 200 períodos
#
# ifr = indice de força relativa
# Índice de força relativa baseado em 21, 50, 80 e 200 períodos
colunas_dados = []
for n in [21, 50, 80, 200]:
    dados_acao_dos_ultimos_10_anos['mma ' + str(n)] = talib.SMA(dados_acao_dos_ultimos_10_anos['Adj Close'].values, timeperiod=n)
    dados_acao_dos_ultimos_10_anos['ifr ' + str(n)] = talib.RSI(dados_acao_dos_ultimos_10_anos['Adj Close'].values, timeperiod=n)

    # Adicionando aos dados de treinamento
    colunas_dados = colunas_dados + ['mma ' + str(n), 'ifr ' + str(n)]


# Calculando mudança percentual do volume de negociações
dados_acao_dos_ultimos_10_anos['Volume_1d_change'] = dados_acao_dos_ultimos_10_anos['Volume'].pct_change()

# Adicionando aos dados de treinamento
coluna_variacao_de_volume_diario = ['Volume_1d_change']
colunas_dados.extend(coluna_variacao_de_volume_diario)


# Calculando o preço futuro com base nos valores passados
# 5d_future_close: representa o preço de fechamento ajustado da ação 5 dias no futuro
# 5d_close_future_pct: representa a variação percentual no preço ajustado nos últimos 5 dias
# em relação ao preço futuro de 5 dias.
# A função pct_change(5) calcula a variação percentual em relação a 5 dias atrás.
# dados_acao_dos_ultimos_10_anos['5d_future_close'] = dados_acao_dos_ultimos_10_anos['Adj Close'].shift(-5)
# dados_acao_dos_ultimos_10_anos['5d_close_future_pct'] = dados_acao_dos_ultimos_10_anos['Adj Close'].pct_change(5)
dados_acao_dos_ultimos_10_anos['1d_future_close'] = dados_acao_dos_ultimos_10_anos['Adj Close'].shift(-1)
dados_acao_dos_ultimos_10_anos['1d_close_future_pct'] = dados_acao_dos_ultimos_10_anos['Adj Close'].pct_change(1)


# Visualizando os dados em tabela dos últimos 5 dias de negociação
# dados_acao_dos_ultimos_10_anos.tail()


# Removendo valores nulos da nossa tabela de dados para treinamento
dados_acao_dos_ultimos_10_anos.dropna(inplace=True)


# Preparando dados para modelagem
# X representa as características usadas para prever a variável alvo.

# y representa a variável alvo que você está tentando prever, neste caso,
# '5d_close_future_pct', que é a mudança percentual no preço ajustado nos
# últimos 5 dias em relação ao preço futuro de 5 dias.
X = dados_acao_dos_ultimos_10_anos[colunas_dados]
# y = dados_acao_dos_ultimos_10_anos['5d_close_future_pct']
y = dados_acao_dos_ultimos_10_anos['1d_close_future_pct']



# Separando os dados em conjuntos de treinamento e teste para a construção e avaliação do modelo
percentual_total_de_amostras = (85 / 100) # Em ML é comum usar 85% das amostras para treino (mas poderia ser 80%, 90%, etc)
numero_de_amostras = y.shape[0] # Shape retorna o número total de elementos em y
tamanho_conjunto_treinamento = int(percentual_total_de_amostras * numero_de_amostras)

# Separando
X_treino = X[:tamanho_conjunto_treinamento]
y_treino = y[:tamanho_conjunto_treinamento]
X_teste = X[tamanho_conjunto_treinamento:]
y_teste = y[tamanho_conjunto_treinamento:]


# # Buscando melhor combinação de hiperparametros
# # Criando hiperparâmetros para treinamento do modelo Random Forest
# # n_estimators: Vai testar o modelo com 200 estimadores (árvores) na floresta.
# # max_depth: Vai testar o modelo com uma profundidade máxima de árvore igual a 3.
# # max_features: Vai testar o modelo com 4 e 8 características máximas em cada divisão.
# # random_state: Fixa a semente aleatória (número usado para inicializar o gerador de números aleatórios em um programa de computador) para garantir a reprodutibilidade dos resultados.
hiperparametros_random_forest = {'n_estimators': [500], 'max_depth': [10], 'max_features': [4, 8], 'random_state': [42]}
resultados_testes = []

# Utilizando o método RandomForestRegressor e ParameterGrid da lib "sklearn" para
# treinar o modelo baseado nos conjuntos que separamos acima.
modelo_random_forest = RandomForestRegressor()

for parametro in ParameterGrid(hiperparametros_random_forest):
    modelo_random_forest.set_params(**parametro)
    modelo_random_forest.fit(X_treino, y_treino)
    resultados_testes.append(modelo_random_forest.score(X_teste, y_teste))

maior_pontuacao = np.argmax(resultados_testes)
print("\n Maior pontuação: ")
print(resultados_testes[maior_pontuacao], ParameterGrid(hiperparametros_random_forest)[maior_pontuacao])
print("\n ")

# Treinando modelo com hiperparâmetros testados no trecho comentado acima.
modelo_random_forest = RandomForestRegressor(n_estimators=500, max_depth=10, max_features=8, random_state=42)
modelo_random_forest.fit(X_treino, y_treino)

previsao_preco_fechamento = modelo_random_forest.predict(X_teste)
serie_previsao_preco_fechamento = pd.Series(previsao_preco_fechamento, index=y_teste.index)

serie_previsao_preco_fechamento.plot()
plt.ylabel("Percentual de Mudança Previsto no Preço de Fechamento em 1 Dia")
plt.show()


print("\n ")


fig = go.Figure()
fig.add_trace(go.Scatter(x=y_teste.index, y=y_teste, mode='lines', name='Preço Real'))
fig.add_trace(go.Scatter(x=serie_previsao_preco_fechamento.index, y=serie_previsao_preco_fechamento, mode='lines', name='Preço Previsto'))
fig.update_layout(title='Preço Real vs. Preço Previsto',
                  xaxis_title='Data',
                  yaxis_title='Percentual de Mudança Previsto no Preço de Fechamento em 1 Dia')
fig.show()


print("\n ")


# Métricas de erro
print('Erro Médio Absoluto (Representa a média do erro absoluto entre as previsões do modelo e os valores reais):', metrics.mean_absolute_error(y_teste, previsao_preco_fechamento))
print('Erro Médio Quadrático (Média dos quadrados das diferenças entre as previsões e os valores verdadeiros):', metrics.mean_squared_error(y_teste, previsao_preco_fechamento))
print('Raiz do Erro Médio Quadrático (Representa a raiz quadrada da média dos quadrados das diferenças entre as previsões e os valores verdadeiros):', np.sqrt(metrics.mean_squared_error(y_teste, previsao_preco_fechamento)))

print("\n ")

importancia_relativa_de_cada_coluna_do_modelo = modelo_random_forest.feature_importances_
indice_ordenado_das_importancias = np.argsort(importancia_relativa_de_cada_coluna_do_modelo)[::-1]

valores_importancias = range(len(importancia_relativa_de_cada_coluna_do_modelo))
labels = np.array(colunas_dados)[indice_ordenado_das_importancias]

plt.bar(valores_importancias, importancia_relativa_de_cada_coluna_do_modelo[indice_ordenado_das_importancias], tick_label=labels)
plt.xticks(rotation=90)
plt.xlabel("Importância relativa de cada indicador utilizado no modelo")
plt.show()