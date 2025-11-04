# src/utils/bet_sizing.py
import yfinance as yf

class BetSizing:
    def __init__(self, ticker):
        self.ticker = ticker
        # Le capital et risk_pct seront passés en argument
        # self.leverage = 30 # Le levier n'est pas utilisé dans position_size_with_atr

    def getlastPrice(self):
        """Récupère le dernier prix minute pour le ticker."""
        try:
            data = yf.download(self.ticker, period="1d", interval='1m', progress=False)['Close']
            if data.empty:
                # Fallback au dernier prix jour
                data = yf.download(self.ticker, period="5d", interval='1d', progress=False)['Close']
            
            last_price = float(data.iloc[-1])
            return last_price
        except Exception as e:
            print(f"Erreur getlastPrice pour {self.ticker}: {e}")
            return None

    def position_size_with_atr(self, capital, risk_pct, entry_price, atr_value, atr_mult=2):
        """
        Calcule la taille de position basée sur l'ATR.
        """
        if entry_price is None or entry_price == 0:
            return 0, 0
            
        # Si l'ATR est en % (log_return std), le convertir en $
        if atr_value < 0.1: # Heuristique pour détecter un ATR en %
            atr_abs = atr_value * entry_price
        else:
            atr_abs = atr_value # Supposer que l'ATR est déjà en $ (si > 10%)

        if atr_abs == 0:
            atr_abs = entry_price * 0.01 # Fallback à 1% du prix

        stop_price = entry_price - atr_mult * atr_abs
        risk_amount = capital * risk_pct # Risque en $ par trade
        risk_per_share = abs(entry_price - stop_price)

        if risk_per_share == 0:
            return 0, stop_price # Ne peut pas diviser par zéro

        shares = risk_amount / risk_per_share
        
        # Gérer la contrainte de capital (ne pas utiliser plus que le capital total)
        max_shares_capital = capital / entry_price
        shares = min(shares, max_shares_capital)
        
        # Vous pourriez aussi vouloir un levier ici, ex:
        # max_shares_lev = (capital * levier) / entry_price
        # shares = min(shares, max_shares_lev)

        return int(shares), stop_price