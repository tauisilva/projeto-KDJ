from modules.lsd import loadStockData
from prophet import Prophet

appleStockHistory = loadStockData("aapl")
print(appleStockHistory.tail(5)) # Últimos 5 dias de negociação
