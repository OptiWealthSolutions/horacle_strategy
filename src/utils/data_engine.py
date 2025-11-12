# src/utils/data_engine.py
import yfinance as yf
import pandas as pd
import numpy as np

class DataEngineer:
    def __init__(self, ticker, interval="1d", period="25y"):
        self.ticker = ticker
        self.PERIOD = period
        self.INTERVAL = interval
        self.SHIFT = 4 # Uniquement pertinent pour les données '1d' (LTF)
    
    def getDataLoad(self):
        data = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL, progress=False)
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = data.dropna()
        
        if data.empty:
            print(f"Avertissement: Aucune donnée {self.INTERVAL} téléchargée pour {self.ticker}.")
            return data

        # Nettoyage des outliers (méthode IQR)
        Q1 = data['Close'].quantile(0.25)
        Q3 = data['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)]
        
        if self.INTERVAL == "1d":
            data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

            data['return'] = data['Close'].pct_change(self.SHIFT).shift(-self.SHIFT)
            data.dropna(inplace=True)

        return data
    def fetchMacroData():
        pass