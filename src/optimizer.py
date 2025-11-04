# src/utils/optimizer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from utils.label_engine import PurgedKFold
from setting import primary_model_params, meta_model_params, embargo # Import des settings

class StrategyOptimizer:
    def __init__(self, model_type='primary', n_splits=5):
        if model_type not in ['primary', 'meta']:
            raise ValueError("model_type doit être 'primary' ou 'meta'")
        self.model_type = model_type
        self.cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo)
        self.scaler = StandardScaler()
        self.best_params_ = None
        
        if model_type == 'primary':
            self.model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
            self.param_grid = primary_model_params
        else:
            self.model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1)
            self.param_grid = meta_model_params

    def optimize(self, X, y, sample_weights=None):
        """
        Effectue une recherche par grille (GridSearch) avec PurgedKFold.
        """
        print(f"Début de l'optimisation pour le modèle {self.model_type}...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 'sample_weight' est un paramètre de fit, nous devons le passer à GridSearchCV
        fit_params = {}
        if sample_weights is not None:
             # Doit s'assurer que les poids sont alignés avec les indices de fold
             # C'est complexe avec GridSearchCV. Une approche plus simple est de 
             # passer les poids alignés sur X_scaled.
             fit_params['sample_weight'] = sample_weights.reindex(X.index).fillna(0).values

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv.split(X_scaled, y), # Passer le générateur
            scoring='f1_macro', # Utiliser f1_macro pour les classes déséquilibrées
            n_jobs=-1, # Utiliser tous les cœurs
            verbose=2
        )
        
        try:
            if sample_weights is not None:
                grid_search.fit(X_scaled, y, **fit_params)
            else:
                 grid_search.fit(X_scaled, y)
                 
            self.best_params_ = grid_search.best_params_
            print(f"Optimisation terminée. Meilleurs paramètres: {self.best_params_}")
            print(f"Meilleur score (f1_macro): {grid_search.best_score_}")
            
            # Retourne le meilleur estimateur déjà entraîné
            return grid_search.best_estimator_, self.best_params_
            
        except Exception as e:
            print(f"Erreur durant GridSearchCV: {e}")
            # Retourner un modèle par défaut en cas d'échec
            default_model = self.model.set_params(**self.param_grid[0] if isinstance(self.param_grid, list) else list(self.param_grid.values())[0][0])
            if sample_weights is not None:
                default_model.fit(X_scaled, y, **fit_params)
            else:
                default_model.fit(X_scaled, y)
            return default_model, None