import yfinance as yf
import pandas as pd
import numpy as np
from src.setting import HTF_interval, LTF_interval,HTF_Start_Date,LTF_Start_Date, End_Date


class DataEngineer():
    def __init__(self):
        ticker = float
        pass

    def getDataLoad(self):
        data = yf.download(self.ticker, period=self.PERIOD, interval=self.INTERVAL, progress= False)
        data = data.dropna()
        Q1 = data['Close'].quantile(0.25)
        Q3 = data['Close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)]
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['return'] = data['Close'].pct_change(self.SHIFT).shift(-self.SHIFT)
        data.dropna(inplace=True)
        return data
    
    def getPCA(self):
        pass