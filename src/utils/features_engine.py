import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from sklearn.metrics import accuracy_score, f1_score


class PrimaryFeaturesEngineer:
    def __init__(self):
        pass

    def getRSI(self, data):
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data
    
    def PriceMomentum(self, data):
        data['PriceMomentum'] = (data['Close'] / data['Close'].shift(12) - 1) * 100
        return data
    
    def getLagReturns(self, data, lags=[12]):
        for n in lags:
            data[f'RETURN_LAG_{n}'] = np.log(data['Close'] / data['Close'].shift(n))
        return data
    
    def PriceAccel(self, data):
        # 'log_return' doit être calculé dans data_engine
        if 'log_return' in data.columns:
            data['velocity'] = data['log_return']
            data['acceleration'] = data['log_return'].diff()
        return data
    
    def getPct52WeekHigh(self, data):
        w_high = data['High'].rolling(window=252).max()
        data['Pct52WeekHigh'] = data['Close'] / w_high
        return data
    
    def getPct52WeekLow(self, data):
        w_low = data['Low'].rolling(window=252).min()
        data['Pct52WeekLow'] = data['Close'] / w_low
        return data
    
    def get12MonthPriceMomentum(self, data):
        data['12MonthPriceMomentum'] = (data['Close'] / data['Close'].shift(252) - 1) * 100
        return data
    
    def getVol(self, data):
        data['MonthlyVol'] = data['Close'].pct_change().rolling(window=20).std()
        return data
        
    def getMacroData(self, data, period="25y"):
        
        # 'data.index' est maintenant garanti "naive" grâce à data_engine.py
        
        # Télécharger DXY via yfinance
        try:
            dxy = yf.download("DX-Y.NYB", period=period, interval="1d", progress=False)['Close']
            
            # --- CORRECTION 1: Standardiser DXY ---
            if dxy.index.tz is not None:
                dxy.index = dxy.index.tz_localize(None) 
            
            data['DXY'] = dxy.reindex(data.index, method='ffill')
            
        except Exception as e:
            print(f"Warn: Impossible de télécharger DXY: {e}")
            data['DXY'] = np.nan # Mettre NaN en cas d'échec

        # Télécharger TWI via FRED
        try:
            # pandas_datareader a besoin de dates simples (str)
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            
            twi = web.DataReader("DTWEXBGS", "fred", start=start_date, end=end_date)
            twi = twi.resample("D").last() 

            # FRED est déjà 'naive', pas besoin de convertir le TZ
            data['TWI'] = twi['DTWEXBGS'].reindex(data.index, method='ffill')
            
        except Exception as e:
            print(f"Warn: Impossible de télécharger TWI (FRED): {e}")
            data['TWI'] = np.nan # Mettre NaN en cas d'échec
        
        # --- CORRECTION 2: Gérer les NaNs de manière robuste ---
        # Remplir les NaNs partiels (bfill/ffill)
        data['DXY'] = data['DXY'].ffill().bfill() 
        data['TWI'] = data['TWI'].ffill().bfill() 

        # Remplir les NaNs restants (colonnes entièrement vides) avec 0
        # pour empêcher le dropna() de strategy.py de tout supprimer.
        data['DXY'] = data['DXY'].fillna(0)
        data['TWI'] = data['TWI'].fillna(0)
                
        return data

    def getFeaturesDataSet(self, data):
        # Supprime les colonnes inutiles pour le modèle
        features_to_drop = ['High', 'Low', 'Open', 'Volume', 'Close', 'return', 'velocity',
                            'label_entry_date', 'label_exit_date', 'label_entry_price',
                            'label_exit_price', 'label_return', 'label_hold_days',
                            'label_barrier_hit', 'vol_adjustment', 'Target', 'SampleWeight',
                            'primary_signal', 'meta_signal'] # Etre exhaustif
        
        data_features = data.drop(columns=features_to_drop, axis=1, errors='ignore')
        return data_features


class MetaFeaturesEngineer:
    def __init__(self):
        pass

    def getEntropy(self, meta_data_proba):
        probabilities = meta_data_proba.values
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        return pd.Series(entropy, index=meta_data_proba.index)

    def getMaxProbability(self, meta_data_proba):
        max_probs = np.max(meta_data_proba.values, axis=1)
        return pd.Series(max_probs, index=meta_data_proba.index)
    
    def getMarginConfidence(self, meta_data_proba):
        probs = meta_data_proba.values
        sorted_probs = np.sort(probs, axis=1)
        # Gérer le cas où il y a moins de 2 classes (ex: 1 classe)
        if sorted_probs.shape[1] < 2:
            return pd.Series(sorted_probs[:, -1], index=meta_data_proba.index)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # Plus haute - 2ème plus haute
        return pd.Series(margin, index=meta_data_proba.index)
    
    def getF1Scoredata(self, y_true, y_pred, window_size=50):
        rolling_f1 = []
        for i in range(len(y_pred)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            if end_idx - start_idx >= 10: # Besoin d'assez de données pour un score stable
                window_f1 = f1_score(
                    y_true[start_idx:end_idx],
                    y_pred[start_idx:end_idx],
                    average='macro',
                    zero_division=0
                )
            else:
                window_f1 = 0.0
            rolling_f1.append(window_f1)
        return pd.Series(rolling_f1, index=y_true.index)

    def getAccuracydata(self, y_true, y_pred, window_size=50):
        rolling_acc = []
        for i in range(len(y_pred)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            if end_idx - start_idx >= 10:
                window_acc = accuracy_score(
                    y_true[start_idx:end_idx],
                    y_pred[start_idx:end_idx]
                )
            else:
                window_acc = 0.0
            rolling_acc.append(window_acc)
        return pd.Series(rolling_acc, index=y_true.index)