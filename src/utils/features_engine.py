import pandas as pd 
import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score, f1_score

class PrimaryFeaturesEngineer():
    def __init__(self):
        pass
    def getRSI(self):
        self.data['RSI'] = self.data['Close'].diff().pipe(lambda x: x.clip(lower=0)).ewm(alpha=1/14, adjust=False).mean() / self.data['Close'].diff().pipe(lambda x: -x.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        self.data['RSI'] = 100 - (100 / (1 + self.data['RSI']))
        return self.data
    
    def PriceMomentum(self):
        self.data['PriceMomentum'] = (self.data['Close'] / self.data['Close'].shift(12) - 1) * 100
        return self.data
    
    def getLagReturns(self):
        for n in self.lags:
            self.data[f'RETURN_LAG_{n}'] = np.log(self.data['Close'] / self.data['Close'].shift(n))
        return self.data
    
    def PriceAccel(self):
        self.data['velocity'] = self.data['log_return']
        self.data['acceleration'] = self.data['log_return'].diff()    
        return self.data
    
    def getPct52WeekHigh(self):
        w_high = self.data['High'].rolling(window=252).max()
        self.data['Pct52WeekHigh'] = self.data['Close'] / w_high
        return self.data
    
    def getPct52WeekLow(self):
        w_low = self.data['Low'].rolling(window=252).min()
        self.data['Pct52WeekLow'] = self.data['Close'] / w_low
        return self.data
    
    def get12MonthPriceMomentum(self):
        self.data['12MonthPriceMomentum'] = (self.data['Close'] / self.data['Close'].shift(252) - 1) * 100
        return self.data
    
    def getVol(self):
        self.data['MonthlyVol'] = self.data['Close'].pct_change().rolling(window=20).std()
        return self.data
        
    def getMacroData(self):
        import pandas_datareader.data as web

        # Télécharger DXY et VIX via yfinance
        dxy = yf.download("DX-Y.NYB", period=self.PERIOD, interval="1d",progress= False)['Close']

        # Télécharger TWI via FRED
        try:
            twi = web.DataReader("DTWEXBGS", "fred")
            twi = twi.resample("D").last()
            twi = twi['DTWEXBGS']
        except:
            twi = pd.Series(index=self.data.index, data=np.nan)

        # Réindexer et forward-fill
        self.data['DXY'] = dxy.reindex(self.data.index, method='ffill')
        self.data['TWI'] = twi.reindex(self.data.index, method='ffill')
        return self.data

    # --- Dataset Preparation ---
    def getFeaturesDataSet(self):
        self.data_features = self.data.drop(['High', 'Low', 'Open', 'Volume', 'Close', 'Return', 'Velocity'], axis=1, errors='ignore')
        return self.data_features

        pass

    
class MetaFeaturesEngineer():
    def __init__(self):
        pass
    # --- Meta Features ---
    def getEntropy(self):
        probabilities = self.meta_data.values
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        self.meta_features_data['prediction_entropy'] = entropy
        return

    def getMaxProbability(self):
        max_probs = np.max(self.meta_data.values, axis=1)
        self.meta_features_data['max_probability'] = max_probs 
        return 
    
    def getMarginConfidence(self):
        probs = self.meta_data.values
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # Plus haute - 2ème plus haute
        self.meta_features_data['margin_confidence'] = margin
        return margin
    
    def getF1Scoredata(self, y_true, y_pred, window_size=50):
        rolling_f1 = []
        for i in range(len(y_pred)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            if end_idx - start_idx >= 10:
                window_f1 = f1_score(
                    y_true[start_idx:end_idx],
                    y_pred[start_idx:end_idx],
                    average='macro'
                )
            else:
                window_f1 = 0.0
            rolling_f1.append(window_f1)
        self.meta_features_data['f1_score'] = rolling_f1
        return rolling_f1

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
        self.meta_features_data['accuracy'] = rolling_acc
        return rolling_acc

    def getMetaFeaturesdata(self):
        return self.meta_features_data


    