from datetime import datetime, timedelta

# clé api et MT5 server informations
# .env

# --- Environnement de Trading ---
# Choisissez "DEMO" pour les tests ou "LIVE" pour le trading réel
ENVIRONMENT="DEMO"

# --- MetaTrader 5 Credentials ---
MT5_LOGIN_DEMO=12345678
MT5_PASSWORD_DEMO="votre_mot_de_passe_demo"
MT5_SERVER_DEMO="VotreBroker-Demo"

MT5_LOGIN_LIVE=87654321
MT5_PASSWORD_LIVE="votre_VRAI_mot_de_passe"
MT5_SERVER_LIVE="VotreBroker-Live"

# --- Autres Clés API (ex: pour des données news) ---
NEWS_API_KEY="votre_cle_api_news"
fred_api =  ""


# données du pf
capital = 1000

# start date / end date
HTF_Start_Date = "2000-01-01"
LTF_Start_Date = "2025-01-01"
End_Date = datetime.today().date() - timedelta(days=2)

# yfinance/vector bt interval and period
HTF_interval = "1d"
LTF_interval = "1h"

#risk params
exposition_max = 0.1*capital #il faut actualiser avec le vrai montant du capital recuperer via MT5
riskMax_trade = 0.02*capital
leverage = 30


# grid seaach params and tunning params
primary_model_params = {
    "n_estimators" : [50,100,200,300],  
    "max_depth" : [5,10,15,20],
}


meta_model_params = {
    "n_estimators" : [50,100,200,300],
    "max_depth" : [5,10,15,20],
}

#embargo
embargo = 0.01

# test tickers and validation tickers
# --- Listes de test (petit échantillon pour débogage rapide) ---
test_FX_tickers = [
    "EURUSD=X",  # Euro / Dollar US
    "GBPUSD=X",  # Livre / Dollar US
    "USDJPY=X",  # Dollar US / Yen
    "USDCAD=X"   # Dollar US / Dollar canadien
]

test_equity_tickers = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "SPY",    # ETF S&P 500
    "GLD"     # Or (ETF)
]

# --- Listes complètes (production) ---
fx_tickers = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCAD=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "USDCHF=X",
    "EURGBP=X",
    "EURJPY=X",
    "GBPJPY=X"
]

equity_tickers = [
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "DAX",    # Indice allemand
    "FTSE",   # Indice britannique
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "TSLA",   # Tesla
    "GLD",    # Or (commodity proxy)
    "USO"     # Pétrole brut (commodity proxy)
]

#plotly and backtest visualization params
