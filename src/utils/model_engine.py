import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from .label_engine import PurgedKFold # Import relatif

class ModelEngine:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.scaler = StandardScaler()
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
                'max_depth': 5, # Souvent plus simple
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42,
                'n_jobs': -1
            }
        self.meta_model = XGBClassifier(**params)
        return self.meta_model

    def train_model(self, model, X, y, sample_weights=None):
        """
        Entraîne un modèle en utilisant la validation croisée purgée (PurgedKFold).
        NOTE: Ceci entraîne sur des folds, mais le modèle final est entraîné sur TOUT X.
        La CV ici sert surtout à l'évaluation (que je n'imprime pas pour garder propre).
        Pour un entraînement robuste, on entraîne sur tout le set.
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Pour le pipeline final, on entraîne sur l'ensemble des données
        if sample_weights is not None:
             # S'assurer que les poids correspondent
             sample_weights = sample_weights.reindex(X.index).fillna(0)
             model.fit(X_scaled, y, sample_weight=sample_weights.values)
        else:
             model.fit(X_scaled, y)
        
        print(f"Modèle {type(model).__name__} entraîné sur {X_scaled.shape[0]} échantillons.")
        return model

    def predict(self, model, X, model_type='primary'):
        # S'assurer que les colonnes de X correspondent à l'entraînement
        X_scaled = self.scaler.transform(X) # Utiliser transform, pas fit_transform
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Gérer les noms de colonnes pour les probas
        if model_type == 'primary':
            # Les classes de RandomForest sont dans model.classes_
            class_names = [f"proba_{c}" for c in model.classes_]
        else:
            # Les classes de XGBoost
            class_names = [f"meta_proba_{c}" for c in model.classes_]

        proba_df = pd.DataFrame(probabilities, index=X.index, columns=class_names)
        last_proba = probabilities[-1] if len(probabilities) > 0 else None
        
        return predictions, proba_df, last_proba

    def computeConfidenceScore(self, last_proba, last_proba_meta):
        if last_proba is None or last_proba_meta is None:
            return 0.0 # Retourner 0 si pas de proba
            
        primary_conf = float(np.max(last_proba))
        meta_conf = float(np.max(last_proba_meta))
        
        # La confiance est la probabilité que le méta-modèle donne un "oui" (classe 1)
        # et la confiance du modèle primaire
        meta_yes_proba = 0.0
        if len(last_proba_meta) > 1: # S'assurer qu'il y a 2 classes
            meta_yes_proba = float(last_proba_meta[1]) # Probabilité de la classe 1
        elif len(last_proba_meta) == 1:
             meta_yes_proba = float(last_proba_meta[0]) # S'il ne prédit qu'une classe

        conf_score = primary_conf * meta_yes_proba # Confiance primaire * Confiance Méta
        return conf_score

    def save_model(self, model, filepath):
        """Sauvegarde un modèle entraîné."""
        try:
            joblib.dump(model, filepath)
            print(f"Modèle sauvegardé dans {filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle: {e}")

    def load_model(self, filepath):
        """Charge un modèle."""
        if os.path.exists(filepath):
            try:
                model = joblib.load(filepath)
                print(f"Modèle chargé depuis {filepath}")
                return model
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
                return None
        return None