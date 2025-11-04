import vectorbt as vbt
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
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
from label_engine import PurgedKFold


#intern function and class : 
from features_engine import PrimaryFeaturesEngineer, MetaFeaturesEngineer
from label_engine import PrimaryLabel, MetaLabel
# --- PDF reportlab imports ---


class PrimaryModel():
    def __init__():
        pass

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
    
