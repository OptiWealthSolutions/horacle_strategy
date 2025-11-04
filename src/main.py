


class Strategy():
    def __init__():

        pass
    def main(self):
        ms = 1
        ms.getRSI()
        ms.PriceMomentum()
        ms.getLagReturns()
        ms.PriceAccel()
        ms.getPct52WeekLow()
        ms.getPct52WeekHigh()
        ms.getVol()
        ms.getMacroData()
        ms.getFeaturesDataSet()
        ms.getLabels()
        ms.getSampleWeight()
        ms.PrimaryModel()
        ms.getEntropy()
        ms.getMaxProbability()
        ms.getMarginConfidence()
        ms.getF1Scoredata(ms.data['Target'], ms.primary_predictions)
        ms.getAccuracydata(ms.data['Target'], ms.primary_predictions)
        ms.getMetaFeaturesdata()
        ms.metaLabeling()
        ms.MetaModel()
        conf_score = ms.computeConfidenceScore()
        bs = BetSizing(ms.ticker)
        last_price = bs.getlastPrice()
        capital = 885
        risk_pct = 0.01
        if 'log_return' in ms.data.columns:
            atr_value = ms.data['log_return'].rolling(14).std().iloc[-1]
        else:
            atr_value = 0.01
        shares, stop = bs.position_size_with_atr(capital, risk_pct, last_price, atr_value)
        ms.data['atr_value'] = atr_value
        summary_data = summarize_signal(ms, shares, stop, last_price, capital, risk_pct, conf_score)
        return ms, summary_data

    if __name__ == "__main__":
        results = {}
        summaries = []
        for ticker in ticker_list:
            try:
                ms, summary_data = main(ticker)
                results[ticker] = ms
                if summary_data is not None and not summary_data.empty:
                    summaries.append(summary_data)
            except Exception as e:
                print(f"Erreur sur {ticker}: {e}")
        if summaries:
            data_summary = pd.concat(summaries, ignore_index=True)
            print("\n=== SIGNAL SUMMARY ===")
            print(data_summary)
            all_signals_path = "all_signals.csv"
            data_summary.to_csv(all_signals_path, index=False)
            print(f"All signals saved to {all_signals_path}")
            try:
                pdf_path = "summary_signals.pdf"
                doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                elements = []
                styles = getSampleStyleSheet()
                title = Paragraph("Trading Signal Summary Report", styles["Heading1"])
                elements.append(title)
                elements.append(Spacer(1, 12))
                table_data = [list(data_summary.columns)] + data_summary.values.tolist()
                table = Table(table_data)
                table_style = TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ])
                table.setStyle(table_style)
                elements.append(table)
                doc.build(elements)
                print(f"PDF summary saved as: {pdf_path}")
            except Exception as e:
                print(f"Erreur lors de la génération du PDF: {e}")
        else:
            print("\nNo valid signals to summarize.")

