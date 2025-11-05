# src/utils/model_engine.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from .label_engine import PurgedKFold # Import relatif
from sklearn.base import clone # <-- NOUVEL IMPORT

class ModelEngine:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.primary_scaler = StandardScaler() 
        self.meta_scaler = StandardScaler()    
        self.cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
        self.primary_model = None
        self.meta_model = None

    def build_primary_model(self, params=None):
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        self.primary_model = RandomForestClassifier(**params)
        return self.primary_model
        
    def build_meta_model(self, params=None):
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 5, 
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42,
                'n_jobs': -1
            }
        self.meta_model = XGBClassifier(**params)
        return self.meta_model

    # --- FONCTION train_model ENTIÈREMENT MISE À JOUR ---
    def train_model(self, model, X, y, sample_weights=None, model_type='primary'):
        if model_type == 'primary':
            scaler = self.primary_scaler
        else: # 'meta'
            scaler = self.meta_scaler
            
        # 1. Fitter le scaler sur TOUTES les données X
        X_scaled = scaler.fit_transform(X)
        
        # 2. Boucle de validation croisée (CV) pour obtenir les scores
        print(f"Calcul des scores (CV={self.cv.n_splits} folds) pour le modèle {model_type}...")
        accuracies = []
        f1_scores = []
        
        # S'assurer que y et sample_weights sont des Series/DF pour .iloc
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        if sample_weights is not None and not isinstance(sample_weights, pd.Series):
            sample_weights = pd.Series(sample_weights, index=X.index)

        for train_idx, test_idx in self.cv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Cloner le modèle pour un entraînement propre à ce fold
            fold_model = clone(model)
            
            fit_params = {}
            if sample_weights is not None:
                 fit_params['sample_weight'] = sample_weights.iloc[train_idx].values
                 fold_model.fit(X_train, y_train, **fit_params)
            else:
                 fold_model.fit(X_train, y_train)
            
            # Prédiction et calcul des scores
            y_pred = fold_model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        avg_acc = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)

        # 3. Entraînement du modèle final sur TOUTES les données
        print(f"Entraînement du modèle final {model_type} sur {X_scaled.shape[0]} échantillons...")
        if sample_weights is not None:
             model.fit(X_scaled, y, sample_weight=sample_weights.values)
        else:
             model.fit(X_scaled, y)
        
        print(f"Modèle {type(model).__name__} entraîné.")
        
        # 4. Retourner le modèle fitté et les scores
        return model, avg_acc, avg_f1
    # --- FIN DE LA MISE À JOUR ---

    def predict(self, model, X, model_type='primary'):
        if model_type == 'primary':
            scaler = self.primary_scaler
        else: 
            scaler = self.meta_scaler
        
        X_scaled = scaler.transform(X) 
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        if model_type == 'primary':
            class_names = [f"proba_{c}" for c in model.classes_]
        else:
            class_names = [f"meta_proba_{c}" for c in model.classes_]

        proba_df = pd.DataFrame(probabilities, index=X.index, columns=class_names)
        last_proba = probabilities[-1] if len(probabilities) > 0 else None
        
        return predictions, proba_df, last_proba

    def computeConfidenceScore(self, last_proba, last_proba_meta):
        if last_proba is None or last_proba_meta is None:
            return 0.0
        primary_conf = float(np.max(last_proba))
        meta_yes_proba = 0.0
        if len(last_proba_meta) > 1:
            meta_yes_proba = float(last_proba_meta[1])
        elif len(last_proba_meta) == 1:
             meta_yes_proba = float(last_proba_meta[0])
        conf_score = primary_conf * meta_yes_proba
        return conf_score

    def save_model(self, model, filepath, model_type='primary'):
        scaler_path = filepath.replace("_model.joblib", "_scaler.joblib")
        try:
            joblib.dump(model, filepath)
            print(f"Modèle sauvegardé dans {filepath}")
            if model_type == 'primary':
                joblib.dump(self.primary_scaler, scaler_path)
            else:
                joblib.dump(self.meta_scaler, scaler_path)
            print(f"Scaler sauvegardé dans {scaler_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle/scaler: {e}")

    def load_model(self, filepath, model_type='primary'):
        scaler_path = filepath.replace("_model.joblib", "_scaler.joblib")
        if os.path.exists(filepath) and os.path.exists(scaler_path):
            try:
                model = joblib.load(filepath)
                print(f"Modèle chargé depuis {filepath}")
                if model_type == 'primary':
                    self.primary_scaler = joblib.load(scaler_path)
                else:
                    self.meta_scaler = joblib.load(scaler_path)
                print(f"Scaler chargé depuis {scaler_path}")
                return model
            except Exception as e:
                print(f"Erreur lors du chargement du modèle/scaler: {e}")
                return None
        print(f"Fichier modèle ({filepath}) ou scaler ({scaler_path}) non trouvé. Ré-entraînement nécessaire.")
        return None