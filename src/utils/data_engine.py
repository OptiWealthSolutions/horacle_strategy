# src/utils/data_engine.py
import yfinance as yf
import pandas as pd
import numpy as np

class DataEngineer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.PERIOD = "25y"  # Période de 25 ans
        self.INTERVAL = "1d" # Intervalle d'un jour
        self.SHIFT = 4       # Shift pour le calcul du 'return'
    
    def getDataLoad(self):

        data = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL, progress=False)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        # --- FIN DE LA CORRECTION ---

        data = data.dropna()
        
        if data.empty:
            print(f"Avertissement: Aucune donnée téléchargée pour {self.ticker} après le premier dropna().")
            return data # Retourner le DF vide si le téléchargement échoue

        # Nettoyage des outliers (méthode IQR)
        Q1 = data['Close'].quantile(0.25)
        Q3 = data['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)]
        
        # Calculs de base
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        # 'return' est le retour futur sur SHIFT périodes, utilisé pour le 'Target' initial
        data['return'] = data['Close'].pct_change(self.SHIFT).shift(-self.SHIFT)
        
        data.dropna(inplace=True)
        return data