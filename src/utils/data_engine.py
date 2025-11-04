import yfinance as yf
import pandas as pd
import numpy as np

# Les variables de setting.py seront passées par la classe Strategy
# ou peuvent être importées directement si vous préférez.
# Pour l'instant, je garde les valeurs de momentum_strat.py.

class DataEngineer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.PERIOD = "25y"  # Période de 25 ans
        self.INTERVAL = "1d" # Intervalle d'un jour
        self.SHIFT = 4       # Shift pour le calcul du 'return'
    
    def getDataLoad(self):
        data = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL, progress=False)
        data = data.dropna()
        
        # Nettoyage des outliers (méthode IQR de momentum_strat.py)
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