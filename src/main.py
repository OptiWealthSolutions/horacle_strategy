# src/main.py
import pandas as pd
import warnings
from datetime import datetime
import os 
# Importer les tickers depuis les settings
from setting import test_FX_tickers, fx_tickers

# Importer la classe Stratégie principale
from strategy import Strategy

# Importer les utilitaires de backtest et de reporting
from utils.backtest_engine import BacktestEngine
from utils.reporting import generate_pdf_report

warnings.filterwarnings('ignore')

def run_strategy_for_tickers(ticker_list, force_retrain=False, optimize=False, run_backtests=True):
    all_summaries = []
    all_backtest_stats = []
    
    for ticker in ticker_list:
        try:
            # 1. Initialiser la stratégie pour le ticker
            strat = Strategy(ticker)
            
            # 2. Exécuter le pipeline complet (data, features, models)
            pipeline_success = strat.run_pipeline(
                force_retrain=force_retrain, 
                optimize_models=optimize
            )
            
            if not pipeline_success:
                print(f"Échec du pipeline pour {ticker}. Passage au suivant.")
                continue
                
            # 3. Obtenir le signal de trading actuel
            summary_data = strat.get_trade_signal()
            
            if summary_data is not None and not summary_data.empty:
                all_summaries.append(summary_data)

            # 4. (Optionnel) Exécuter le backtest
            if run_backtests:
                print(f"Exécution du backtest pour {ticker}...")
                bt = BacktestEngine(strat.data) # Passer les données avec signaux
                stats = bt.run_backtest()
                if stats is not None:
                    all_backtest_stats.append((ticker, stats))
                    # bt.plot_backtest(ticker) # Décommenter pour voir les graphiques
                
        except Exception as e:
            print(f"--- ERREUR MAJEURE sur {ticker}: {e} ---")

    return all_summaries, all_backtest_stats


if __name__ == "__main__":
    
    # --- Configuration de l'exécution ---
    
    # Mettre à True pour forcer le ré-entraînement de tous les modèles
    FORCE_RETRAIN_MODELS = True
    
    # Mettre à True pour lancer le (long) processus d'optimisation (GridSearch)
    OPTIMIZE_MODELS = False
    
    # Mettre à True pour calculer les backtests (peut être long)
    RUN_BACKTESTS = False
    
    # Choisir la liste de tickers
    tickers_to_run = ["EURAUD=X","EURNZD=X","USDNZD=X","USDAUD=X"]# Pour un test rapide
    #tickers_to_run = fx_tickers # Pour la "production"
    
    print(f"Démarrage de l'exécution pour {len(tickers_to_run)} tickers...")
    
    # --- Exécution ---
    summaries, backtest_stats = run_strategy_for_tickers(
        tickers_to_run,
        force_retrain=FORCE_RETRAIN_MODELS,
        optimize=OPTIMIZE_MODELS,
        run_backtests=RUN_BACKTESTS
    )

    # --- Génération du Rapport ---
    if summaries:
        final_summary_df = pd.concat(summaries, ignore_index=True)
        print("\n=== RÉSUMÉ GLOBAL DES SIGNAUX ===")
        print(final_summary_df)
        
        # Sauvegarder en CSV
        csv_path = "all_signals.csv"
        final_summary_df.to_csv(csv_path, index=False)
        print(f"Signaux sauvegardés dans {csv_path}")
        
    else:
        print("\n=== AUCUN SIGNAL ACTIF TROUVÉ ===")
        final_summary_df = pd.DataFrame() # DF vide pour le PDF

    # Générer le PDF
    pdf_path = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    generate_pdf_report(final_summary_df, backtest_stats, pdf_path=pdf_path)

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True) # <-- AJOUTER CECI
    
    print(f"Démarrage de l'exécution pour {len(tickers_to_run)} tickers...")
    print("\n--- Exécution terminée ---")