# src/strategy.py
import pandas as pd
import numpy as np
import os
import joblib

# Import des settings et des moteurs utilitaires
from setting import capital, riskMax_trade
from utils.data_engine import DataEngineer
from utils.features_engine import PrimaryFeaturesEngineer, MetaFeaturesEngineer
from utils.label_engine import PrimaryLabel, MetaLabel
from utils.model_engine import ModelEngine
from utils.risk_engine import BetSizing
from optimizer import StrategyOptimizer
from utils.reporting import summarize_signal
from analysis_engine import AnalysisEngine
class Strategy:
    def __init__(self, ticker, model_dir="models"):
        self.ticker = ticker
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # --- Moteurs ---
        self.data_engine = DataEngineer(self.ticker)
        self.feature_engine = PrimaryFeaturesEngineer()
        self.label_engine = PrimaryLabel()
        self.model_engine = ModelEngine()
        self.meta_feature_engine = MetaFeaturesEngineer()
        self.meta_label_engine = MetaLabel()
        self.bet_sizer = BetSizing(self.ticker)

        # --- Dataframes ---
        self.data = pd.DataFrame()
        self.data_features = pd.DataFrame()
        self.meta_data_proba = pd.DataFrame()
        self.meta_features = pd.DataFrame() 
        self.meta_labels = pd.DataFrame() 

        # --- Artefacts ---
        self.primary_model = None
        self.meta_model = None
        self.primary_predictions = None
        self.last_proba = None
        self.last_proba_meta = None
        self.conf_score = None
        
        # --- NOUVEAU : Stockage des scores ---
        self.primary_acc = None
        self.primary_f1 = None
        self.meta_acc = None
        self.meta_f1 = None


    def get_model_path(self, model_type):
        return os.path.join(self.model_dir, f"{self.ticker}_{model_type}_model.joblib")

    def run_pipeline(self, force_retrain=False, optimize_models=False):
        print(f"\n--- Exécution du pipeline pour {self.ticker} ---")
        
        # ... (Étapes 1, 2, 3 inchangées) ...
        # 1. Chargement données...
        self.data = self.data_engine.getDataLoad()
        if self.data.empty:
            print(f"Aucune donnée chargée pour {self.ticker}. Arrêt.")
            return False

        # 2. Features Primaires...
        print("Calcul des features primaires...")
        self.data = self.feature_engine.getRSI(self.data)
        self.data = self.feature_engine.PriceMomentum(self.data)
        self.data = self.feature_engine.getLagReturns(self.data, lags=[12])
        self.data = self.feature_engine.PriceAccel(self.data)
        self.data = self.feature_engine.getPct52WeekLow(self.data)
        self.data = self.feature_engine.getPct52WeekHigh(self.data)
        self.data = self.feature_engine.getVol(self.data)
        self.data = self.feature_engine.getMacroData(self.data, self.data_engine.PERIOD)
        
        self.data_features = self.feature_engine.getFeaturesDataSet(self.data)
        self.data = self.data.dropna()
        self.data_features = self.data_features.reindex(self.data.index).dropna()
        
        if self.data.empty or self.data_features.empty:
            print(f"Données vides après feature engineering pour {self.ticker}. Arrêt.")
            return False

        # 3. Labels Primaires...
        print("Calcul des labels primaires...")
        self.data = self.label_engine.getLabels(self.data)
        self.data = self.data[self.data['Target'] != 0].copy()
        print(f"Distribution des classes après filtrage:\n{self.data['Target'].value_counts()}")
        self.data['SampleWeight'] = self.label_engine.getSampleWeight(
            labels=self.data['Target'], 
            features=self.data_features, 
            timestamps=self.data.index,
            label_endtimes=self.data['label_exit_date']
        )
        self.data_features = self.data_features.reindex(self.data.index)
        self.data = self.data.reindex(self.data_features.index)

        # 4. Modèle Primaire
        print("Traitement du modèle primaire...")
        primary_model_path = self.get_model_path('primary')
        
        if not force_retrain and not optimize_models:
            self.primary_model = self.model_engine.load_model(primary_model_path, model_type='primary')
            if self.primary_model is None:
                 force_retrain = True
        
        if force_retrain or optimize_models or self.primary_model is None:
            if optimize_models:
                optimizer = StrategyOptimizer(model_type='primary')
                # --- MODIFICATION : Capturer les 3 retours ---
                self.primary_model, self.primary_acc, self.primary_f1 = optimizer.optimize(
                    self.data_features, 
                    self.data['Target'], 
                    self.data['SampleWeight']
                )
                self.model_engine.primary_scaler = optimizer.scaler
            else:
                print("Entraînement du modèle primaire (par défaut)...")
                self.primary_model = self.model_engine.build_primary_model()
                # --- MODIFICATION : Capturer les 3 retours ---
                self.primary_model, self.primary_acc, self.primary_f1 = self.model_engine.train_model(
                    self.primary_model, 
                    self.data_features, 
                    self.data['Target'], 
                    self.data['SampleWeight'],
                    model_type='primary'
                )
            
            # Affichage des scores (seulement si entraîné/optimisé)
            print(f"  > Modèle Primaire - Précision CV: {self.primary_acc:.2%}")
            print(f"  > Modèle Primaire - F1 Score CV: {self.primary_f1:.4f}")

            self.model_engine.save_model(self.primary_model, primary_model_path, model_type='primary')
        
        # Prédiction
        self.primary_predictions, self.meta_data_proba, self.last_proba = self.model_engine.predict(
            self.primary_model, 
            self.data_features,
            model_type='primary'
        )
        self.data['primary_signal'] = self.primary_predictions

        # 5. Meta Features
        print("Calcul des méta-features...")
        self.meta_features['prediction_entropy'] = self.meta_feature_engine.getEntropy(self.meta_data_proba)
        self.meta_features['max_probability'] = self.meta_feature_engine.getMaxProbability(self.meta_data_proba)
        self.meta_features['margin_confidence'] = self.meta_feature_engine.getMarginConfidence(self.meta_data_proba)
        self.meta_features['f1_score'] = self.meta_feature_engine.getF1Scoredata(self.data['Target'], self.primary_predictions)
        self.meta_features['accuracy'] = self.meta_feature_engine.getAccuracydata(self.data['Target'], self.primary_predictions)
        
        # 6. Meta Label
        self.meta_labels = self.meta_label_engine.metaLabeling(
            self.primary_predictions, 
            self.data['label_return']
        )
        self.data['meta_label'] = self.meta_labels
        
        self.meta_features = self.meta_features.dropna()
        self.meta_labels = self.meta_labels.reindex(self.meta_features.index)
        
        if self.meta_features.empty:
            print(f"Méta-features vides pour {self.ticker}. Arrêt.")
            return False

        # 7. Meta Modèle
        print("Traitement du méta-modèle...")
        meta_model_path = self.get_model_path('meta')

        if not force_retrain and not optimize_models:
            self.meta_model = self.model_engine.load_model(meta_model_path, model_type='meta')
            if self.meta_model is None:
                force_retrain = True
                
        if force_retrain or optimize_models or self.meta_model is None:
            if optimize_models:
                meta_optimizer = StrategyOptimizer(model_type='meta')
                # --- MODIFICATION : Capturer les 3 retours ---
                self.meta_model, self.meta_acc, self.meta_f1 = meta_optimizer.optimize(
                    self.meta_features, 
                    self.meta_labels
                )
                self.model_engine.meta_scaler = meta_optimizer.scaler
            else:
                print("Entraînement du méta-modèle (par défaut)...")
                self.meta_model = self.model_engine.build_meta_model()
                # --- MODIFICATION : Capturer les 3 retours ---
                self.meta_model, self.meta_acc, self.meta_f1 = self.model_engine.train_model(
                    self.meta_model, 
                    self.meta_features, 
                    self.meta_labels,
                    model_type='meta'
                )
            
            # Affichage des scores (seulement si entraîné/optimisé)
            print(f"  > Méta-Modèle - Précision CV: {self.meta_acc:.2%}")
            print(f"  > Méta-Modèle - F1 Score CV: {self.meta_f1:.4f}")
            
            self.model_engine.save_model(self.meta_model, meta_model_path, model_type='meta')
        
        # Prédiction Méta
        meta_preds, _, self.last_proba_meta = self.model_engine.predict(
            self.meta_model, 
            self.meta_features,
            model_type='meta'
        )
        self.data['meta_signal'] = pd.Series(meta_preds, index=self.meta_features.index)
        self.data['meta_signal'] = self.data['meta_signal'].fillna(0)
        self.data['meta_signal'] = pd.Series(meta_preds, index=self.meta_features.index)
        self.data['meta_signal'] = self.data['meta_signal'].fillna(0) # 0 par défaut

        # 8. Calcul du score de confiance final
        self.conf_score = self.model_engine.computeConfidenceScore(self.last_proba, self.last_proba_meta)
        print(f"Pipeline terminé pour {self.ticker}. Score de confiance: {self.conf_score:.2%}")
        
        
        # --- SECTION D'ANALYSE AJOUTÉE ---
        # 9. Générer le rapport d'analyse détaillé
        try:
            analyzer = AnalysisEngine(self.ticker)
            
            # Aligner les données pour l'analyse
            
            # 1. Données Primaires (alignées sur data_features)
            aligned_primary_true = self.data.loc[self.data_features.index, 'Target']
            aligned_primary_pred = self.data.loc[self.data_features.index, 'primary_signal']
            
            # 2. Données Méta (alignées sur meta_features)
            aligned_meta_true = self.meta_labels.reindex(self.meta_features.index)
            # 'meta_preds' est un np.array, il faut le convertir en Série avec le bon index
            aligned_meta_pred = pd.Series(meta_preds, index=self.meta_features.index)

            analyzer.generate_analysis_report(
                primary_y_true=aligned_primary_true,
                primary_y_pred=aligned_primary_pred,
                meta_y_true=aligned_meta_true,
                meta_y_pred=aligned_meta_pred
            )
        except Exception as e:
            print(f"Erreur lors de la génération du rapport d'analyse: {e}")

        # 8. Calcul du score de confiance final
        self.conf_score = self.model_engine.computeConfidenceScore(self.last_proba, self.last_proba_meta)
        print(f"Pipeline terminé pour {self.ticker}. Score de confiance: {self.conf_score:.2%}")
        return True

    def get_trade_signal(self):
        if self.conf_score is None:
            print("Aucun score de confiance, exécutez d'abord le pipeline.")
            return pd.DataFrame()

        print("Calcul du signal de trading...")
        last_price = self.bet_sizer.getlastPrice()
        if last_price is None:
            print("Impossible de récupérer le dernier prix.")
            return pd.DataFrame()
            
        current_capital = capital 
        risk_pct = riskMax_trade / capital
        
        if 'log_return' in self.data.columns and not self.data['log_return'].empty:
            atr_value = self.data['log_return'].rolling(14).std().iloc[-1]
        else:
            atr_value = 0.01
        
        if 'atr_value' not in self.data.columns:
            self.data['atr_value'] = np.nan
        self.data.loc[self.data.index[-1], 'atr_value'] = atr_value
        
        shares, stop = self.bet_sizer.position_size_with_atr(
            current_capital, 
            risk_pct,
            last_price, 
            atr_value
        )
        
        summary_data = summarize_signal(
            self.data,
            self.ticker, 
            shares, 
            stop, 
            last_price, 
            current_capital, 
            risk_pct,
            self.conf_score
        )
        return summary_data