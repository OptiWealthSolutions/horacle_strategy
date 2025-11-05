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
        print(f"Début de l'optimisation pour le modèle {self.model_type}...")
        
        X_scaled = self.scaler.fit_transform(X) # Fitter le scaler ici
        
        fit_params = {}
        if sample_weights is not None:
             fit_params['sample_weight'] = sample_weights.reindex(X.index).fillna(0).values

        # --- MODIFICATION : Suivre les deux scores ---
        scoring = {'f1': 'f1_macro', 'accuracy': 'accuracy'}
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv.split(X_scaled, y),
            scoring=scoring,       # <-- Modifié
            refit='f1',            # <-- Modifié (choisit le meilleur modèle basé sur f1)
            n_jobs=-1,
            verbose=2,
            return_train_score=False # Gagner du temps
        )
        
        try:
            grid_search.fit(X_scaled, y, **fit_params)
                 
            self.best_params_ = grid_search.best_params_
            
            # --- MODIFICATION : Extraire les deux scores ---
            best_index = grid_search.best_index_
            best_f1 = grid_search.cv_results_['mean_test_f1'][best_index]
            best_acc = grid_search.cv_results_['mean_test_accuracy'][best_index]
            
            print(f"Optimisation terminée.")
            print(f"  Meilleurs Paramètres: {self.best_params_}")
            print(f"  Meilleur F1 Score (CV): {best_f1:.4f}")
            print(f"  Meilleure Précision (CV): {best_acc:.2%}")
            
            # Retourner le meilleur estimateur, sa précision, et son f1 score
            return grid_search.best_estimator_, best_acc, best_f1
            
        except Exception as e:
            print(f"Erreur durant GridSearchCV: {e}")
            # Retourner un modèle par défaut en cas d'échec
            default_model = self.model.set_params(**self.param_grid[0] if isinstance(self.param_grid, list) else list(self.param_grid.values())[0][0])
            default_model.fit(X_scaled, y, **fit_params)
            return default_model, 0.0, 0.0 # Retourner scores nuls