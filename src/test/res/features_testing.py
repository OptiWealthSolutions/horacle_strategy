import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
ticker = "EURUSD=X"
data = yf.download(ticker,period='20y',interval='1d')

def getYieldSpread( data, ticker):
        """
        Calcule l'écart de taux (yield spread) en utilisant UNIQUEMENT l'API FRED
        pour les obligations à 10 ans.
        """
        if ticker is None:
            return data
            
        try:
            # Dictionnaire des tickers FRED pour les taux 10 ans
            # Source : https://fred.stlouisfed.org/
            fred_yield_tickers = {
                'US_10Y': 'DGS10',          # US 10-Year Treasury
                'EUR_10Y': 'IRLTLT01DEM156N', # German 10-Year (Proxy pour l'Euro)
                'GBP_10Y': 'IRLTLT01GBM156N', # UK 10-Year
                'JPY_10Y': 'IRLTLT01JPM156N', # Japan 10-Year
                'CAD_10Y': 'IRLTLT01CAM156N'  # Canada 10-Year
            }
            
            # Définir la plage de dates à partir de l'index de nos données
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            
            # Télécharger toutes les données de taux de FRED en une seule fois
            yield_data = web.DataReader(
                list(fred_yield_tickers.values()), 
                'fred', 
                start_date, 
                end_date
            )

            # Renommer les colonnes avec des noms clairs (ex: 'DGS10' -> 'US_10Y')
            reverse_map = {v: k for k, v in fred_yield_tickers.items()}
            yield_data = yield_data.rename(columns=reverse_map)

            # Les données FRED sont "naives" (sans fuseau horaire), ce qui est 
            # cohérent avec la correction que nous avons apportée à data_engine.py.

            # Aligner sur l'index de 'data', combler les jours manquants (weekends, etc.)
            # C'est une étape cruciale.
            yield_data = yield_data.reindex(data.index, method='ffill').ffill().bfill()

            # Ajouter le taux US 10Y comme feature (toujours utile)
            data['US_10Y_Yield'] = yield_data['US_10Y']

            # Calculer le spread spécifique au ticker
            # Note : C'est Taux_Devise_Base - Taux_Devise_Quote
            # Pour USDJPY, JPY est la devise "quote", donc c'est US - JPY
            
            data['Ticker_Yield_Spread'] = 0.0 # Initialiser
            
            if "EUR" in ticker: # EURUSD
                data['Ticker_Yield_Spread'] = yield_data['EUR_10Y'] - yield_data['US_10Y']
            elif "GBP" in ticker: # GBPUSD
                data['Ticker_Yield_Spread'] = yield_data['GBP_10Y'] - yield_data['US_10Y']
            elif "JPY" in ticker: # USDJPY
                data['Ticker_Yield_Spread'] = yield_data['US_10Y'] - yield_data['JPY_10Y']
            elif "CAD" in ticker: # USDCAD
                # Attention : CAD est la devise quote, donc c'est US - CAD
                data['Ticker_Yield_Spread'] = yield_data['US_10Y'] - yield_data['CAD_10Y']
            
            # Remplir les NaNs restants (si un taux n'a pas pu être téléchargé)
            data['Ticker_Yield_Spread'] = data['Ticker_Yield_Spread'].fillna(0)
            data['US_10Y_Yield'] = data['US_10Y_Yield'].fillna(0)

            print("Features de Yield Spread (FRED) ajoutées.")
            return data

        except Exception as e:
            print(f"Avertissement: Impossible de calculer le Yield Spread (FRED) pour {ticker}. Erreur: {e}")
            # Retourner les données sans les nouvelles features en cas d'échec
            data['Ticker_Yield_Spread'] = 0.0
            data['US_10Y_Yield'] = 0.0
            return data
        
getYieldSpread(data, ticker)

plt.scatter(data['Close'], data['Ticker_Yield_Spread'])
plt.show()