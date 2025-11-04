
import vectorbt as vbt
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import os
# --- PDF reportlab imports ---
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        test_size = n_samples // self.n_splits
        embargo = int(n_samples * self.embargo_pct)
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size
            test_idx = np.arange(test_start, test_end)
            train_idx = np.arange(0, test_start)
            if test_end + embargo < n_samples:
                train_idx = np.concatenate([train_idx, np.arange(test_end + embargo, n_samples)])
            yield train_idx, test_idx
        return

class SampleWeights:
    def __init__(self, labels, features, timestamps):
        self.timestamps = pd.Series(timestamps, index=timestamps)
        self.labels = pd.Series(labels, index=timestamps)
        self.features = features
        self.n_samples = len(labels)
        self.data = pd.DataFrame(features, index=timestamps)
        self.data['labels'] = self.labels

    def getIndMatrix(self, label_endtimes=None):
        if label_endtimes is None:
            label_endtimes = self.timestamps
        molecules = label_endtimes.index
        all_ranges = [(start, label_endtimes[start]) for start in molecules]
        all_times = pd.date_range(self.timestamps.min(), self.timestamps.max(), freq='D')
        indicator = np.zeros((len(molecules), len(all_times)), dtype=np.uint8)
        time_pos = {dt: idx for idx, dt in enumerate(all_times)}
        for sample_idx, (start, end) in enumerate(all_ranges):
            if pd.isna(start) or pd.isna(end):
                continue
            rng = pd.date_range(start, end, freq='D')
            valid_idx = [time_pos[dt] for dt in rng if dt in time_pos]
            if valid_idx:
                indicator[sample_idx, valid_idx] = 1
        indicator[indicator.sum(axis=1) == 0, 0] = 1
        return pd.DataFrame(indicator, index=molecules, columns=all_times)

    def getAverageUniqueness(self, indicator_matrix):
        timestamp_usage_count = indicator_matrix.sum(axis=0).values
        mask = indicator_matrix.values.astype(bool)
        uniqueness_matrix = np.divide(
            mask,
            timestamp_usage_count,
            out=np.zeros_like(mask, dtype=float),
            where=timestamp_usage_count > 0
        )
        avg_uniqueness = uniqueness_matrix.sum(axis=1) / (mask.sum(axis=1) + 1e-10)
        return pd.Series(avg_uniqueness, index=indicator_matrix.index)

    def getRarity(self):
        returns = self.data['labels']
        abs_returns = returns.abs()
        if abs_returns.sum() == 0:
            return pd.Series(np.ones(len(returns))/len(returns), index=returns.index)
        return abs_returns / abs_returns.sum()

    def getSequentialBootstrap(self, indicator_matrix, sample_length=None, random_state=42, n_simulations=10000):
        np.random.seed(random_state)
        n_samples = indicator_matrix.shape[0]
        if sample_length is None:
            sample_length = n_samples
        avg_uniqueness = self.getAverageUniqueness(indicator_matrix)
        probabilities = avg_uniqueness / avg_uniqueness.sum()
        all_choices = np.random.choice(
            n_samples,
            size=n_simulations * sample_length,
            replace=True,
            p=probabilities.values
        ).reshape(n_simulations, sample_length)
        counts = np.bincount(all_choices.ravel(), minlength=n_samples)
        sample_weights = pd.Series(counts, index=indicator_matrix.index)
        sample_weights /= sample_weights.sum() if sample_weights.sum() > 0 else 1
        return sample_weights

    def getRecency(self, decay=0.01):
        time_delta = (self.timestamps.max() - self.timestamps).dt.days
        weights = np.exp(-decay * time_delta)
        return pd.Series(weights, index=self.timestamps.index) / weights.sum()



class MomentumStrategy:
    def __init__(self, ticker):
        self.ticker = ticker
        self.PERIOD = "25y"
        self.INTERVAL = "1d"
        self.SHIFT = 4
        self.lags = [12]
        self.data = self.getDataLoad()
        self.data_features = pd.DataFrame()
        self.meta_data = pd.DataFrame()
        self.meta_features_data = pd.DataFrame()
        self.last_proba = None
        self.last_proba_meta = None


    # --- Data Loading ---
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

    # --- Feature Engineering ---
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


    # --- Labeling / Weights ---
    def getLabels(self, max_hold_days=12, stop_loss=0.01, profit_target=0.05, volatility_scaling=True):
        prices = self.data['Close']
        n = len(prices)
        prices_array = prices.values
        labels = np.zeros(n)
        entry_dates, exit_dates, entry_prices, exit_prices = [], [], [], []
        returns_pct, hold_days, barrier_hit, vol_adj_arr = [], [], [], []

        def _find_first_barrier_hit(prices, entry_idx, profit_target, stop_loss, max_hold):
            entry_price = prices[entry_idx]
            end_idx = min(entry_idx + max_hold, len(prices) - 1)
            for i in range(entry_idx + 1, end_idx + 1):
                raw_ret = (prices[i] - entry_price) / entry_price
                if raw_ret >= profit_target:
                    return 1, i  # Profit hit
                elif raw_ret <= -stop_loss:
                    return -1, i  # Stop loss hit
            return 0, end_idx  # Time barrier hit

        if volatility_scaling:
            returns = prices.pct_change()
            vol = returns.rolling(20).std().fillna(returns.std())
            vol_filled = vol.fillna(method='bfill').fillna(method='ffill')
        else:
            vol_filled = None  # Pas utilisé

        for i in range(n):
            if np.isnan(prices_array[i]):
                labels[i] = 0
                entry_dates.append(prices.index[i])
                exit_dates.append(prices.index[i])
                entry_prices.append(prices_array[i])
                exit_prices.append(prices_array[i])
                returns_pct.append(0)
                hold_days.append(0)
                barrier_hit.append('NaN')
                vol_adj_arr.append(1.0)
                continue

            if volatility_scaling:
                vol_value = float(vol_filled.iloc[i])
                vol_adj = max(vol_value / 0.02, 0.5)
                profit_adj = profit_target * vol_adj
                loss_adj = stop_loss * vol_adj
            else:
                profit_adj = profit_target
                loss_adj = stop_loss
                vol_adj = 1.0

            label, exit_idx = _find_first_barrier_hit(
                prices_array, i, profit_adj, loss_adj, max_hold_days
            )

            labels[i] = label
            entry_dates.append(prices.index[i])
            exit_dates.append(prices.index[exit_idx])
            entry_prices.append(prices_array[i])
            exit_prices.append(prices_array[exit_idx])
            raw_return = (prices_array[exit_idx] - prices_array[i]) / prices_array[i]
            returns_pct.append(raw_return)
            hold_days.append(exit_idx - i)
            barrier_hit.append(['Time', 'Profit', 'Loss'][label + 1])
            vol_adj_arr.append(vol_adj)

        self.data['Target'] = labels
        self.data['label_entry_date'] = entry_dates
        self.data['label_exit_date'] = exit_dates
        self.data['label_entry_price'] = entry_prices
        self.data['label_exit_price'] = exit_prices
        self.data['label_return'] = returns_pct
        self.data['label_hold_days'] = hold_days
        self.data['label_barrier_hit'] = barrier_hit
        self.data['vol_adjustment'] = vol_adj_arr

        return self.data
    
    def getSampleWeight(self, decay=0.01):
        sw = SampleWeights(labels=self.data['Target'], features=self.data_features, timestamps=self.data.index)
        label_endtimes = None
        if 'label_exit_date' in self.data.columns:
            label_endtimes = self.data['label_exit_date']
        indicator_matrix = sw.getIndMatrix(label_endtimes=label_endtimes)
        rarity_weights = sw.getRarity()
        recency_weights = sw.getRecency(decay)
        sequential_weights = sw.getSequentialBootstrap(indicator_matrix)
        common_index = rarity_weights.index.intersection(recency_weights.index).intersection(sequential_weights.index)
        combined = (
            rarity_weights.loc[common_index].fillna(0) *
            recency_weights.loc[common_index].fillna(0) *
            sequential_weights.loc[common_index].fillna(0)
        )
        if combined.sum() > 0:
            combined = combined / combined.sum()
        else:
            combined = pd.Series(np.ones(len(self.data.index)) / len(self.data.index), index=self.data.index)
        full_weights = combined.reindex(self.data.index).fillna(0)
        self.data['SampleWeight'] = full_weights
        return full_weights
    
    # --- Primary Model ---
    def PrimaryModel(self, n_splits=5):
        if not hasattr(self, 'data_features') or self.data_features.empty:
            self.getFeaturesDataSet()
        X = self.data_features.values
        y = self.data['Target'].values
        sample_weights = self.data['SampleWeight'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tscv = PurgedKFold(n_splits=n_splits, embargo_pct=0.01)
        scores = []
        reports = []
        cms = []
        f1_score_ = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weights[train_idx]
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            cms.append(confusion_matrix(y_test, y_pred))
            f1_score_.append(f1_score(y_test, y_pred, average='weighted'))
        primary_preds = model.predict(X_scaled)
        self.data['primary_signal'] = primary_preds
        last_pred = primary_preds[-1]
        last_proba = model.predict_proba(X_test)[-1]
        self.meta_data = pd.DataFrame(model.predict_proba(X_scaled), index=self.data.index)
        self.primary_predictions = primary_preds
        self.last_proba = last_proba
        self.last_pred = last_pred
        return self.meta_data, last_proba

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

    # --- Meta Labelling ---
    def metaLabeling(self):
        model_predictions = self.primary_predictions != 0  # True si le modèle a généré un signal
        actual_profitable = self.data['label_return'] > 0     # True si le trade était profitable
        
        meta_labels = (model_predictions & actual_profitable).astype(int)
        
        self.meta_data = pd.DataFrame(index=self.data.index)
        self.meta_data['meta_label'] = meta_labels
        
        return
    
    # --- Meta Model ---
    def MetaModel(self):
        X = self.meta_features_data.values
        y = self.meta_data['meta_label'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tscv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        scores = []
        reports = []
        cms = []
        f1_score_ = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
            meta_model.fit(X_train, y_train)
            y_pred = meta_model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
            reports.append(classification_report(y_test, y_pred, output_dict=True))
            cms.append(confusion_matrix(y_test, y_pred))
            f1_score_.append(f1_score(y_test, y_pred, average='weighted'))
        high = self.data['High'] if 'High' in self.data.columns else self.data['Close']
        low = self.data['Low'] if 'Low' in self.data.columns else self.data['Close']
        close = self.data['Close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_value = tr.rolling(14).mean().iloc[-1]
        self.meta_preds = meta_model.predict(X_scaled)
        self.data['meta_signal'] = self.meta_preds
        last_pred = self.meta_preds[-1]
        last_proba_meta = meta_model.predict_proba(X_test)[-1]
        self.last_proba_meta = last_proba_meta
        self.last_pred_meta = last_pred
        return meta_model, last_proba_meta

    # --- Confidence Score ---
    def computeConfidenceScore(self):
        if self.last_proba is None or self.last_proba_meta is None:
            return None
        primary_conf = float(np.max(self.last_proba)) if hasattr(self.last_proba, '__len__') else float(self.last_proba)
        meta_conf = float(np.max(self.last_proba_meta)) if hasattr(self.last_proba_meta, '__len__') else float(self.last_proba_meta)
        conf_score = primary_conf * meta_conf
        return conf_score
    
class BetSizing:
    def __init__(self, ticker):
        self.ticker = ticker
        self.capital = None
        self.risk_pct = None
        self.leverage = 30
    def getlastPrice(self):
        data = yf.download(self.ticker, period="1d", interval='1m', progress=False)['Close']
        last_price = float(data.iloc[-1])
        return last_price

    def position_size_with_atr(self, capital, risk_pct, entry_price, atr_value, atr_mult=2):
        if atr_value < 0.1:
            atr_abs = atr_value * entry_price
        else:
            atr_abs = atr_value
        stop_price = entry_price - atr_mult * atr_abs
        risk_amount = capital * risk_pct
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            return 0, stop_price
        shares = risk_amount // risk_per_share
        max_shares = capital // entry_price
        shares = min(shares, max_shares)
        return int(shares), stop_price


class Backtest:
    def __init__(self, ms):
        self.ms = ms
        self.data_backtest = self.ms.data.loc["2012-01-01":"2025-01-01"].copy()
        self.data_backtest['signal'] = 0
        self.data_backtest.loc[(self.data_backtest['primary_signal'] == 1) & (self.data_backtest['meta_signal'] == 1), 'signal'] = 1
        self.data_backtest.loc[(self.data_backtest['primary_signal'] == -1) & (self.data_backtest['meta_signal'] == 1), 'signal'] = -1
        self.entries = self.data_backtest['signal'] == 1
        self.exits = self.data_backtest['signal'] == -1

    def portfolio(self):
        self.pf = vbt.Portfolio.from_signals(
            close=self.data_backtest['Close'],
            entries=self.entries,
            exits=self.exits,
            init_cash=10_000,
            fees=0.005,
            freq="1D"
        )
        self.data_backtest['portfolio_value'] = self.pf.value()
        sharpe_ratio = self.pf.sharpe_ratio()
        self.data_backtest['portfolio_value'].plot()
        plt.show()
        return self.ms



# --- GLOBALS & MAIN ---

ticker_list = [
    # --- Paires majeures (USD base ou contrepartie) ---
    "EURUSD=X",  # Euro / Dollar américain
    "GBPUSD=X",  # Livre sterling / Dollar américain
    "USDCAD=X",  # Dollar américain / Dollar canadien
    "USDCHF=X",  # Dollar américain / Franc suisse
    "USDJPY=X",  # Dollar américain / Yen japonais

    # --- Paires croisées principales ---
    "EURGBP=X",  # Euro / Livre sterling
    "EURCAD=X",  # Euro / Dollar canadien
    "GBPCAD=X",  # Livre sterling / Dollar canadien
    "EURCHF=X",  # Euro / Franc suisse
    "GBPCHF=X",  # Livre sterling / Franc suisse
]

def summarize_signal(ms, shares, stop, last_price, capital, risk_pct, conf_score):
    primary_signal = ms.data['primary_signal'].iloc[-1] if 'primary_signal' in ms.data.columns else 0
    meta_signal = ms.data['meta_signal'].iloc[-1] if 'meta_signal' in ms.data.columns else 0
    signal = None
    if (primary_signal == 1) and (meta_signal == 1):
        signal = "BUY"
    elif (primary_signal == -1) and (meta_signal == 1):
        signal = "SELL"
    if signal is not None:
        row = {
            "ticker": ms.ticker,
            "signal": signal,
            "last_price": last_price,
            "shares": shares,
            "stop": stop,
            "confidence_score": conf_score,
            "risk_amount": risk_pct * capital * conf_score,
            "atr" : ms.data['atr_value'].iloc[-1]
        }
        return pd.DataFrame([row])
    else:
        return pd.DataFrame(columns=["ticker", "signal", "last_price", "shares", "stop", "confidence_score", "risk_amount"])


def main(ticker):
    ms = MomentumStrategy(ticker)
    ms.getRSI()
    ms.PriceMomentum()
    ms.getLagReturns()
    ms.PriceAccel()
    ms.getPct52WeekLow()
    ms.getPct52WeekHigh()
    ms.getVol()
    ms.getMacroData()
    ms.getFeaturesDataSet()
    ms.getLabels()
    ms.getSampleWeight()
    ms.PrimaryModel()
    ms.getEntropy()
    ms.getMaxProbability()
    ms.getMarginConfidence()
    ms.getF1Scoredata(ms.data['Target'], ms.primary_predictions)
    ms.getAccuracydata(ms.data['Target'], ms.primary_predictions)
    ms.getMetaFeaturesdata()
    ms.metaLabeling()
    ms.MetaModel()
    conf_score = ms.computeConfidenceScore()
    bs = BetSizing(ms.ticker)
    last_price = bs.getlastPrice()
    capital = 885
    risk_pct = 0.01
    if 'log_return' in ms.data.columns:
        atr_value = ms.data['log_return'].rolling(14).std().iloc[-1]
    else:
        atr_value = 0.01
    shares, stop = bs.position_size_with_atr(capital, risk_pct, last_price, atr_value)
    ms.data['atr_value'] = atr_value
    summary_data = summarize_signal(ms, shares, stop, last_price, capital, risk_pct, conf_score)
    return ms, summary_data

if __name__ == "__main__":
    results = {}
    summaries = []
    for ticker in ticker_list:
        try:
            ms, summary_data = main(ticker)
            results[ticker] = ms
            if summary_data is not None and not summary_data.empty:
                summaries.append(summary_data)
        except Exception as e:
            print(f"Erreur sur {ticker}: {e}")
    if summaries:
        data_summary = pd.concat(summaries, ignore_index=True)
        print("\n=== SIGNAL SUMMARY ===")
        print(data_summary)
        all_signals_path = "all_signals.csv"
        data_summary.to_csv(all_signals_path, index=False)
        print(f"All signals saved to {all_signals_path}")
        try:
            pdf_path = "summary_signals.pdf"
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            title = Paragraph("Trading Signal Summary Report", styles["Heading1"])
            elements.append(title)
            elements.append(Spacer(1, 12))
            table_data = [list(data_summary.columns)] + data_summary.values.tolist()
            table = Table(table_data)
            table_style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ])
            table.setStyle(table_style)
            elements.append(table)
            doc.build(elements)
            print(f"PDF summary saved as: {pdf_path}")
        except Exception as e:
            print(f"Erreur lors de la génération du PDF: {e}")
    else:
        print("\nNo valid signals to summarize.")

