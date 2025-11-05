# src/utils/analysis_engine.py
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class AnalysisEngine:
    def __init__(self, ticker, report_dir="reports"):
        self.ticker = ticker
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.pdf_path = os.path.join(self.report_dir, f"analyse_performance_{self.ticker}.pdf")
        self.elements = []
        self.styles = getSampleStyleSheet()
        # Style pour le texte pré-formaté (rapport de classification)
        self.pre_style = ParagraphStyle(
            'Preformatted',
            fontName='Courier',
            fontSize=8,
            leading=10,
            leftIndent=6,
            rightIndent=6
        )

    def _save_plot_to_buffer(self, fig):
        """Sauvegarde une figure matplotlib dans un buffer mémoire."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig) # Fermer la figure pour libérer la mémoire
        return buf

    def _convert_plot_to_reportlab_image(self, plot_buffer, width=5.5*inch):
        """Convertit un buffer en image ReportLab."""
        img = Image(plot_buffer)
        img.drawWidth = width
        img.drawHeight = (width / img.imageWidth) * img.imageHeight
        img.hAlign = 'CENTER'
        return img

    def _plot_class_distribution(self, y_true, title):
        """Génère un graphique de distribution des classes."""
        fig, ax = plt.subplots()
        counts = y_true.value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Classe")
        ax.set_ylabel("Nombre d'échantillons")
        return self._save_plot_to_buffer(fig)

    def _plot_confusion_matrix(self, y_true, y_pred, title, labels):
        """Génère une matrice de confusion graphique."""
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vraie valeur")
        return self._save_plot_to_buffer(fig)

    def _generate_classification_report(self, y_true, y_pred, labels):
        """Génère le rapport de classification sous forme de texte."""
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        # Remplacer les espaces par des espaces insécables pour ReportLab
        report = report.replace(" ", "\u00A0")
        report = report.replace("\n", "<br/>")
        return Paragraph(report, self.pre_style)

    def generate_analysis_report(self, primary_y_true, primary_y_pred, meta_y_true, meta_y_pred):
        """
        Orchestre la création du rapport d'analyse complet.
        """
        print(f"Génération du rapport d'analyse pour {self.ticker}...")
        doc = SimpleDocTemplate(self.pdf_path, pagesize=A4)
        
        # --- TITRE ---
        self.elements.append(Paragraph(f"Rapport d'Analyse - {self.ticker}", self.styles['Heading1']))
        self.elements.append(Spacer(1, 0.25*inch))

        # === SECTION MODÈLE PRIMAIRE ===
        self.elements.append(Paragraph("Analyse Modèle Primaire (Label = Target)", self.styles['Heading2']))
        primary_labels = [-1, 0, 1]
        
        # 1. Distribution des classes primaires
        self.elements.append(Paragraph("Distribution des Vraies Classes (Target)", self.styles['Heading3']))
        distrib_primary_plot = self._plot_class_distribution(
            primary_y_true, 
            "Distribution des Labels Primaires (Target)"
        )
        self.elements.append(self._convert_plot_to_reportlab_image(distrib_primary_plot))
        self.elements.append(Spacer(1, 0.2*inch))

        # 2. Rapport de classification primaire
        self.elements.append(Paragraph("Rapport de Classification (Primaire)", self.styles['Heading3']))
        report_primary = self._generate_classification_report(primary_y_true, primary_y_pred, primary_labels)
        self.elements.append(report_primary)
        self.elements.append(Spacer(1, 0.2*inch))

        # 3. Matrice de confusion primaire
        self.elements.append(Paragraph("Matrice de Confusion (Primaire)", self.styles['Heading3']))
        matrix_primary_plot = self._plot_confusion_matrix(
            primary_y_true, 
            primary_y_pred, 
            "Matrice de Confusion Primaire",
            primary_labels
        )
        self.elements.append(self._convert_plot_to_reportlab_image(matrix_primary_plot))

        self.elements.append(PageBreak())

        # === SECTION MÉTA-MODÈLE ===
        self.elements.append(Paragraph("Analyse Méta-Modèle (Label = Profit Oui/Non)", self.styles['Heading2']))
        meta_labels_list = [0, 1]
        
        # 1. Distribution des classes méta
        self.elements.append(Paragraph("Distribution des Vraies Classes (Meta-Label)", self.styles['Heading3']))
        distrib_meta_plot = self._plot_class_distribution(
            meta_y_true, 
            "Distribution des Labels Méta (0=Pas Profit, 1=Profit)"
        )
        self.elements.append(self._convert_plot_to_reportlab_image(distrib_meta_plot))
        self.elements.append(Spacer(1, 0.2*inch))

        # 2. Rapport de classification méta
        self.elements.append(Paragraph("Rapport de Classification (Méta)", self.styles['Heading3']))
        report_meta = self._generate_classification_report(meta_y_true, meta_y_pred, meta_labels_list)
        self.elements.append(report_meta)
        self.elements.append(Spacer(1, 0.2*inch))

        # 3. Matrice de confusion méta
        self.elements.append(Paragraph("Matrice de Confusion (Méta)", self.styles['Heading3']))
        matrix_meta_plot = self._plot_confusion_matrix(
            meta_y_true, 
            meta_y_pred, 
            "Matrice de Confusion Méta",
            meta_labels_list
        )
        self.elements.append(self._convert_plot_to_reportlab_image(matrix_meta_plot))
        
        # --- Génération du PDF ---
        try:
            doc.build(self.elements)
            print(f"Rapport d'analyse sauvegardé dans {self.pdf_path}")
        except Exception as e:
            print(f"Erreur lors de la génération du PDF d'analyse: {e}")