import pandas as pd
import numpy as np


class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        if n_samples == 0:
             raise ValueError("Cannot split empty dataset")
        
        indices = np.arange(n_samples)
        test_size = n_samples // self.n_splits
        embargo = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size
            if i == self.n_splits - 1: # S'assurer que le dernier fold prend tout le reste
                test_end = n_samples
                
            test_idx = indices[test_start:test_end]

            # Indices d'entraînement avant la purge
            train_idx_pre = indices[:test_start]
            # Indices d'entraînement après la purge (avec embargo)
            train_idx_post = indices[test_end + embargo:]
            
            train_idx = np.concatenate([train_idx_pre, train_idx_post])
            
            yield train_idx, test_idx

class SampleWeights:
    def __init__(self, labels, features, timestamps):
        self.timestamps = pd.Series(timestamps, index=timestamps)
        self.labels = pd.Series(labels, index=timestamps)
        self.features = features
        self.n_samples = len(labels)
        self.data = pd.DataFrame(features, index=timestamps)
        if 'labels' not in self.data.columns:
            self.data['labels'] = self.labels

    def getIndMatrix(self, label_endtimes=None):
        if label_endtimes is None:
            label_endtimes = self.timestamps
        
        # S'assurer que label_endtimes est une Série avec le même index que timestamps
        if not isinstance(label_endtimes, pd.Series):
             label_endtimes = pd.Series(label_endtimes, index=self.timestamps.index)
             
        molecules = label_endtimes.index
        all_ranges = [(start, label_endtimes.get(start)) for start in molecules]
        
        # Gérer les NaT potentiels
        all_ranges = [(s, e) for s, e in all_ranges if pd.notna(s) and pd.notna(e)]
        if not all_ranges:
            return pd.DataFrame(index=molecules)

        min_date = min(s for s, e in all_ranges)
        max_date = max(e for s, e in all_ranges)
        
        all_times = pd.date_range(min_date, max_date, freq='D')
        indicator = np.zeros((len(molecules), len(all_times)), dtype=np.uint8)
        time_pos = {dt: idx for idx, dt in enumerate(all_times)}

        for sample_idx, (start, end) in enumerate(all_ranges):
            rng = pd.date_range(start, end, freq='D')
            valid_idx = [time_pos[dt] for dt in rng if dt in time_pos]
            if valid_idx:
                indicator[sample_idx, valid_idx] = 1
        
        # Gérer les échantillons sans plage de temps (par ex. hold_days=0)
        no_time_samples = (indicator.sum(axis=1) == 0)
        if no_time_samples.any():
            indicator[no_time_samples, 0] = 1 # Leur donner au moins une part

        return pd.DataFrame(indicator, index=molecules, columns=all_times)

    def getAverageUniqueness(self, indicator_matrix):
        if indicator_matrix.empty:
            return pd.Series(index=indicator_matrix.index)
            
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
        if indicator_matrix.empty:
             return pd.Series(index=indicator_matrix.index)

        np.random.seed(random_state)
        n_samples = indicator_matrix.shape[0]
        if sample_length is None:
            sample_length = n_samples
        
        avg_uniqueness = self.getAverageUniqueness(indicator_matrix)
        if avg_uniqueness.sum() == 0:
             probabilities = pd.Series(np.ones(n_samples)/n_samples, index=indicator_matrix.index)
        else:
             probabilities = avg_uniqueness / avg_uniqueness.sum()
             
        probabilities = probabilities.fillna(0) # Gérer les NaNs
        probabilities_sum = probabilities.sum()
        if probabilities_sum == 0:
            probabilities = pd.Series(np.ones(n_samples)/n_samples, index=indicator_matrix.index)
        else:
            probabilities /= probabilities_sum


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


class PrimaryLabel:
    def __init__(self):
        pass

    def getLabels(self, data, max_hold_days=5, stop_loss=0.01, profit_target=0.02, volatility_scaling=True):
        prices = data['Close']
        n = len(prices)
        prices_array = prices.values
        labels = np.zeros(n)
        entry_dates, exit_dates, entry_prices, exit_prices = [], [], [], []
        returns_pct, hold_days, barrier_hit, vol_adj_arr = [], [], [], []

        def _find_first_barrier_hit(prices_arr, entry_idx, pt, sl, max_hold):
            entry_price = prices_arr[entry_idx]
            end_idx = min(entry_idx + max_hold, len(prices_arr) - 1)
            
            # Cas où l'entry_price est invalide (évite division par zéro)
            if entry_price == 0:
                return 0, end_idx

            for i in range(entry_idx + 1, end_idx + 1):
                raw_ret = (prices_arr[i] - entry_price) / entry_price
                if raw_ret >= pt:
                    return 1, i  # Profit hit
                elif raw_ret <= -sl:
                    return -1, i  # Stop loss hit
            
            # Si la boucle se termine (Time barrier hit), retourner le label basé sur le prix final
            # et TOUJOURS retourner end_idx
            
            final_ret = (prices_arr[end_idx] - entry_price) / entry_price
            
            if final_ret > 0: 
                return 1, end_idx # Expiration en profit
            elif final_ret < 0: 
                return -1, end_idx # Expiration en perte
            else: 
                return 0, end_idx # Expiration neutre
            

        if volatility_scaling:
            returns = prices.pct_change()
            vol = returns.rolling(20).std().fillna(returns.std())
            vol_filled = vol.fillna(method='bfill').fillna(method='ffill')
        else:
            vol_filled = pd.Series(1.0, index=prices.index) # Pas d'ajustement

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

            vol_adj = 1.0
            if volatility_scaling:
                vol_value = float(vol_filled.iloc[i])
                vol_adj = max(vol_value / 0.02, 0.5) # Logique de momentum_strat.py
            
            profit_adj = profit_target * vol_adj
            loss_adj = stop_loss * vol_adj

            label, exit_idx = _find_first_barrier_hit(
                prices_array, i, profit_adj, loss_adj, max_hold_days
            )

            labels[i] = label
            entry_dates.append(prices.index[i])
            exit_dates.append(prices.index[exit_idx])
            entry_prices.append(prices_array[i])
            exit_prices.append(prices_array[exit_idx])
            
            raw_return = 0.0
            if prices_array[i] != 0: # Eviter division par zéro
                raw_return = (prices_array[exit_idx] - prices_array[i]) / prices_array[i]
                
            returns_pct.append(raw_return)
            hold_days.append(exit_idx - i)
            # Correction: ['Time', 'Profit', 'Loss'][label + 1] -> ['Loss', 'Time', 'Profit'][label + 1]
            barrier_hit.append(['Loss', 'Time', 'Profit'][int(label) + 1])
            vol_adj_arr.append(vol_adj)

        data['Target'] = labels
        data['label_entry_date'] = entry_dates
        data['label_exit_date'] = exit_dates
        data['label_entry_price'] = entry_prices
        data['label_exit_price'] = exit_prices
        data['label_return'] = returns_pct
        data['label_hold_days'] = hold_days
        data['label_barrier_hit'] = barrier_hit
        data['vol_adjustment'] = vol_adj_arr

        return data

    def getSampleWeight(self, labels, features, timestamps, label_endtimes, decay=0.01):
        sw = SampleWeights(labels=labels, features=features, timestamps=timestamps)
        
        indicator_matrix = sw.getIndMatrix(label_endtimes=label_endtimes)
        
        rarity_weights = sw.getRarity()
        recency_weights = sw.getRecency(decay)
        sequential_weights = sw.getSequentialBootstrap(indicator_matrix)

        common_index = rarity_weights.index.intersection(recency_weights.index).intersection(sequential_weights.index)
        
        if common_index.empty:
            return pd.Series(np.ones(len(timestamps)) / len(timestamps), index=timestamps)

        combined = (
            rarity_weights.loc[common_index].fillna(0) *
            recency_weights.loc[common_index].fillna(0) *
            sequential_weights.loc[common_index].fillna(0)
        )
        
        if combined.sum() > 0:
            combined = combined / combined.sum()
        else:
            combined = pd.Series(np.ones(len(common_index)) / len(common_index), index=common_index)
            
        full_weights = combined.reindex(timestamps).fillna(0)
        
        # S'assurer que la somme est 1 si elle n'est pas nulle
        final_sum = full_weights.sum()
        if final_sum > 0:
             return full_weights / final_sum
        else:
             return pd.Series(np.ones(len(timestamps)) / len(timestamps), index=timestamps)


class MetaLabel:
    def __init__(self):
        pass

    def metaLabeling(self, primary_predictions, label_returns):
        """
        Crée les labels pour le méta-modèle.
        Le but est de prédire si un signal du modèle primaire sera profitable.
        Label = 1 si (signal != 0 ET trade profitable)
        Label = 0 sinon
        
        CORRIGÉ : Renvoie une pd.Series avec l'index de label_returns.
        """
        
        # S'assurer que label_returns est une Series pour préserver l'index
        if not isinstance(label_returns, pd.Series):
            try:
                # Si ce n'est pas une Series, on suppose qu'il n'y a pas d'index à sauver
                # (ceci ne devrait pas arriver dans votre pipeline)
                label_returns = pd.Series(label_returns)
            except:
                 raise TypeError("label_returns doit être convertible en pandas.Series.")
        
        # Obtenir les valeurs numpy pour le calcul
        if isinstance(primary_predictions, pd.Series):
            primary_predictions_values = primary_predictions.values
        else:
            primary_predictions_values = primary_predictions # C'est déjà un np.array

        label_returns_values = label_returns.values

        # Logique de calcul (inchangée)
        model_predictions = (primary_predictions_values != 0)  # True si signal
        actual_profitable = (label_returns_values > 0)         # True si profitable
        
        meta_labels_values = (model_predictions & actual_profitable).astype(int)
        
        # --- CORRECTION CLÉ ---
        # Retourner une Series, en utilisant l'index de label_returns pour l'alignement
        meta_labels_series = pd.Series(
            meta_labels_values, 
            index=label_returns.index, 
            name="meta_label"
        )
        
        return meta_labels_series