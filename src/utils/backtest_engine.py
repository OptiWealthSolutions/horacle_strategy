import vectorbt as vbt
import vectorbt as vbt
import matplotlib.pyplot as plt
from setting import LTF_Start_Date, End_Date
class BacktestEngine:
    def __init__(self, data, start_date=LTF_Start_Date, end_date=End_Date):
        self.data_backtest = data.loc[start_date:end_date].copy()
        self.pf = None
        self.stats = None

    def run_backtest(self):
        if 'primary_signal' not in self.data_backtest or 'meta_signal' not in self.data_backtest:
            print("Erreur Backtest: Signaux manquants.")
            return None
            
        self.data_backtest['signal'] = 0
        
        buy_condition = (self.data_backtest['primary_signal'] == 1) & (self.data_backtest['meta_signal'] == 1)
        sell_condition = (self.data_backtest['primary_signal'] == -1) & (self.data_backtest['meta_signal'] == 1)
        
        self.data_backtest.loc[buy_condition, 'signal'] = 1
        self.data_backtest.loc[sell_condition, 'signal'] = -1

        # vectorbt a besoin de signaux d'entrée (1) et de sortie (True/False)
        # entries = (self.data_backtest['signal'] == 1)
        # exits = (self.data_backtest['signal'] == -1)
        
        # Une meilleure approche pour vectorbt est d'utiliser les signaux directionnels
        # 1 = Long, -1 = Short, 0 = Neutre
        
        try:
            self.pf = vbt.Portfolio.from_signals(
                close=self.data_backtest['Close'],
                entries=(self.data_backtest['signal'] == 1),
                exits=(self.data_backtest['signal'] == -1),
                short_entries=(self.data_backtest['signal'] == -1), # Activer la Vente à Découvert
                short_exits=(self.data_backtest['signal'] == 1),  # Sortir du short si signal d'achat
                init_cash=10_000,
                fees=0.005, # 0.5% de frais
                freq="1h"
            )
            
            self.stats = self.pf.stats()
            return self.stats
            
        except Exception as e:
            print(f"Erreur VectorBT: {e}")
            return None

    def plot_backtest(self, ticker=""):
        if self.pf:
            print(f"Affichage du backtest pour {ticker}")
            self.pf.plot(title=f"Backtest de la stratégie pour {ticker}").show()
        else:
            print("Aucun backtest à afficher.")