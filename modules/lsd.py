import os
import pandas as pd
import yfinance as yf

#  Carregando dados da ação
def loadStockData(stockTicker):
    DATA_PATH = "./data/{}.csv".format(stockTicker)

    if os.path.exists(DATA_PATH): 
      with open(DATA_PATH) as file:
        stock_history = pd.read_csv(DATA_PATH)
    else:
        stock = yf.Ticker(stockTicker)
        stock_history = stock.history(period="max")
        stock_history.to_csv(DATA_PATH)
    
    return stock_history
