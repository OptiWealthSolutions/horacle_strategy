import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


class PrimaryLabel():
    def __init__(self):
        pass
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
    



class MetaLabel():
    def __init__(self):
        pass
        # --- Meta Labelling ---
    def metaLabeling(self):
        model_predictions = self.primary_predictions != 0  # True si le modèle a généré un signal
        actual_profitable = self.data['label_return'] > 0     # True si le trade était profitable
        
        meta_labels = (model_predictions & actual_profitable).astype(int)
        
        self.meta_data = pd.DataFrame(index=self.data.index)
        self.meta_data['meta_label'] = meta_labels
        
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
