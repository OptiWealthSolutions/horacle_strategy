# src/utils/reporting.py
import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def summarize_signal(data, ticker, shares, stop, last_price, capital, risk_pct, conf_score):
    """
    Crée une ligne de DataFrame pour le résumé du signal, basé sur la *dernière* ligne de données.
    """
    if data.empty or conf_score is None:
        return pd.DataFrame() # Retourner un DF vide

    primary_signal = data['primary_signal'].iloc[-1] if 'primary_signal' in data.columns else 0
    meta_signal = data['meta_signal'].iloc[-1] if 'meta_signal' in data.columns else 0
    atr_value = data['atr_value'].iloc[-1] if 'atr_value' in data.columns else 0

    signal = "NONE"
    if (primary_signal == 1) and (meta_signal == 1):
        signal = "BUY"
    elif (primary_signal == -1) and (meta_signal == 1):
        signal = "SELL"
    else:
        return pd.DataFrame() # Pas de signal, ne rien ajouter au rapport

    adjusted_risk_pct = risk_pct * conf_score
    risk_amount = capital * adjusted_risk_pct

    row = {
        "Ticker": ticker,
        "Signal": signal,
        "Prix Actuel": f"{last_price:.4f}",
        "Taille (Shares)": shares,
        "Stop Loss": f"{stop:.4f}",
        "Confiance": f"{conf_score:.2%}",
        "Risque ($)": f"{risk_amount:.2f}",
        "ATR (14j)": f"{atr_value:.5f}"
    }
    return pd.DataFrame([row])

def generate_pdf_report(data_summary, backtest_stats_list, pdf_path="summary_signals.pdf"):
    """
    Génère un rapport PDF complet avec le résumé des signaux et les statistiques de backtest.
    """
    try:
        doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4)) # Paysage
        elements = []
        styles = getSampleStyleSheet()

        # --- CORRECTION : Définir table_style ici ---
        # Définir le style de tableau une seule fois, en dehors des conditions
        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        # --- FIN DE LA CORRECTION ---


        # --- Page 1: Résumé des Signaux ---
        title = Paragraph("Rapport des Signaux de Trading Actifs", styles["Heading1"])
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        if not data_summary.empty:
            table_data = [list(data_summary.columns)] + data_summary.values.tolist()
            table = Table(table_data, hAlign='LEFT')
            
            # Appliquer le style
            table.setStyle(table_style)
            elements.append(table)
        else:
            elements.append(Paragraph("Aucun signal de trading actif trouvé.", styles["Normal"]))

        elements.append(PageBreak())

        # --- Pages Suivantes: Statistiques de Backtest ---
        title_bt = Paragraph("Résumé des Backtests (2012-2025)", styles["Heading1"])
        elements.append(title_bt)
        elements.append(Spacer(1, 0.2 * inch))

        if backtest_stats_list:
            bt_summary_df = pd.DataFrame(columns=[
                "Ticker", "Total Return [%]", "Sharpe Ratio", "Sortino Ratio", 
                "Max Drawdown [%]", "Win Rate [%]", "Total Trades"
            ])
            
            for ticker, stats in backtest_stats_list:
                row = {
                    "Ticker": ticker,
                    "Total Return [%]": f"{stats.get('Total Return [%]', 0):.2f}",
                    "Sharpe Ratio": f"{stats.get('Sharpe Ratio', 0):.2f}",
                    "Sortino Ratio": f"{stats.get('Sortino Ratio', 0):.2f}",
                    "Max Drawdown [%]": f"{stats.get('Max Drawdown [%]', 0):.2f}",
                    "Win Rate [%]": f"{stats.get('Win Rate [%]', 0):.2f}",
                    "Total Trades": f"{stats.get('Total Trades', 0)}"
                }
                # Utiliser pd.concat pour éviter l'avertissement 'FutureWarning'
                bt_summary_df = pd.concat([bt_summary_df, pd.DataFrame([row])], ignore_index=True)
                
            bt_table_data = [list(bt_summary_df.columns)] + bt_summary_df.values.tolist()
            bt_table = Table(bt_table_data, hAlign='LEFT', colWidths=[
                1.5*inch, 1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch, 1.2*inch, 1*inch
            ])
            
            # Appliquer le même style
            bt_table.setStyle(table_style) 
            elements.append(bt_table)
            
        else:
            elements.append(Paragraph("Aucune statistique de backtest disponible.", styles["Normal"]))


        doc.build(elements)
        print(f"Rapport PDF sauvegardé sous: {pdf_path}")
    except Exception as e:
        print(f"Erreur lors de la génération du PDF: {e}")