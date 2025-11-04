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

class ModelEngine:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        # AVOIR DEUX SCALERS, UN POUR CHAQUE MODÈLE
        self.primary_scaler = StandardScaler() # Scaler pour le modèle primaire
        self.meta_scaler = StandardScaler()    # Scaler pour le méta-modèle
        
        self.cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
        self.primary_model = None
        self.meta_model = None

    def build_primary_model(self, params=None):
        """Crée le modèle primaire avec des paramètres par défaut ou fournis."""
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
        """Crée le méta-modèle avec des paramètres par défaut ou fournis."""
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

    # --- CORRIGÉ : Ajout de model_type='primary' ---
    def train_model(self, model, X, y, sample_weights=None, model_type='primary'):
        """
        Entraîne un modèle. Utilise le scaler correct et le FIT_TRANSFORM.
        """
        if model_type == 'primary':
            scaler = self.primary_scaler
        else: # 'meta'
            scaler = self.meta_scaler
            
        # Fitter le scaler et transformer les données
        X_scaled = scaler.fit_transform(X)
        
        if sample_weights is not None:
             sample_weights = sample_weights.reindex(X.index).fillna(0)
             model.fit(X_scaled, y, sample_weight=sample_weights.values)
        else:
             model.fit(X_scaled, y)
        
        print(f"Modèle {type(model).__name__} entraîné sur {X_scaled.shape[0]} échantillons.")
        return model

    # --- CORRIGÉ : Ajout de model_type='primary' ---
    def predict(self, model, X, model_type='primary'):
        """
        Effectue des prédictions. Utilise le scaler correct en mode TRANSFORM (non-fit).
        """
        if model_type == 'primary':
            scaler = self.primary_scaler
        else: # 'meta'
            scaler = self.meta_scaler
        
        # Le scaler (primary_scaler ou meta_scaler) doit être fitté
        # (soit par train_model, soit chargé depuis le disque).
        X_scaled = scaler.transform(X) # Utiliser transform, pas fit_transform
        
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

    # --- CORRIGÉ : Ajout de model_type='primary' ---
    def save_model(self, model, filepath, model_type='primary'):
        """Sauvegarde le modèle ET son scaler correspondant."""
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

    # --- CORRIGÉ : Ajout de model_type='primary' ---
    def load_model(self, filepath, model_type='primary'):
        """Charge le modèle ET son scaler correspondant."""
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
        
        # Si un des fichiers manque, on ne charge rien et on forcera le ré-entraînement
        print(f"Fichier modèle ({filepath}) ou scaler ({scaler_path}) non trouvé. Ré-entraînement nécessaire.")
        return None