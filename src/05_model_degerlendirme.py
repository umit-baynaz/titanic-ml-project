# -*- coding: utf-8 -*-
"""
Titanic ML Project - Adım 5: Model Değerlendirme
================================================
En iyi modelin detaylı performans analizi, confusion matrix,
ROC curve ve feature importance grafikleri.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'dark': '#2C3E50',
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#2ECC71', '#9B59B6', '#E74C3C']
}

def load_model_and_data():
    print("=" * 60)
    print("ADIM 5: MODEL DEĞERLENDİRME")
    print("=" * 60)
    
    best_model = joblib.load('models/best_model.pkl')
    model_info = joblib.load('models/model_info.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    df = pd.read_csv('data/processed/ml_features.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    
    print(f"\n📊 Model: {model_info['best_model_name']}")
    print(f"   Test seti: {X_test.shape[0]} satır")
    
    return best_model, model_info, X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

def calculate_metrics(model, X_test, y_test):
    """Detaylı metrikler hesapla."""
    print("\n--- Performans Metrikleri ---")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred)
    }
    
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, 
                                    target_names=['Hayatini Kaybeden', 'Hayatta Kalan'])
    print(report)
    
    return y_pred, metrics

def plot_confusion_matrix(y_test, y_pred):
    """Confusion matrix grafiği."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Hayatini Kaybeden', 'Hayatta Kalan'],
                yticklabels=['Hayatini Kaybeden', 'Hayatta Kalan'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 20, 'fontweight': 'bold'})
    
    ax.set_title('Confusion Matrix (Karisiklik Matrisi)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Tahmin Edilen', fontsize=13)
    ax.set_ylabel('Gercek Deger', fontsize=13)
    
    # Yüzdeleri de ekle
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i][j] / total * 100
            ax.text(j + 0.5, i + 0.72, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=11, color='gray')
    
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ confusion_matrix.png")
    
    # Confusion matrix verisini CSV olarak kaydet
    cm_df = pd.DataFrame(cm, 
                         columns=['Tahmin_Oldu', 'Tahmin_Kaldi'],
                         index=['Gercek_Oldu', 'Gercek_Kaldi'])
    cm_df.to_csv('data/processed/confusion_matrix_data.csv')
    
    return cm

def plot_roc_curve(model, X_test, y_test):
    """ROC curve grafiği."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_test)
    else:
        print("  ⚠ Model probability desteği yok, ROC curve atlanıyor.")
        return
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=3,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])
    ax.plot([0, 1], [0, 1], color=COLORS['danger'], linewidth=2, linestyle='--',
            label='Rastgele Tahmin (AUC = 0.50)', alpha=0.7)
    
    ax.set_title('ROC Curve (Alici Isletim Karakteristigi)', fontsize=16, fontweight='bold')
    ax.set_xlabel('False Positive Rate (Yanlis Pozitif Orani)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Dogru Pozitif Orani)', fontsize=12)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ roc_curve.png (AUC: {roc_auc:.4f})")

def plot_feature_importance(model, feature_names):
    """Feature importance grafiği."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("  ⚠ Model feature importance desteği yok.")
        return
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'],
            color=colors, edgecolor='white', linewidth=1)
    
    ax.set_title('Ozellik Onemliligi (Feature Importance)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Onemilik Skoru', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ feature_importance.png")

def plot_model_comparison():
    """Model karşılaştırma grafiği."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    results_df = pd.read_csv('data/processed/model_comparison.csv')
    results_df = results_df.sort_values('CV_Mean', ascending=True)
    
    colors = [COLORS['palette'][i % len(COLORS['palette'])] for i in range(len(results_df))]
    
    bars = ax.barh(results_df['Model'], results_df['CV_Mean'],
                   xerr=results_df['CV_Std'], color=colors,
                   edgecolor='white', linewidth=1.5, capsize=5)
    
    # Test accuracy noktaları
    ax.scatter(results_df['Test_Accuracy'], results_df['Model'],
               color=COLORS['danger'], s=100, zorder=5, marker='D',
               label='Test Accuracy', edgecolors='white', linewidths=1.5)
    
    ax.set_title('Model Karsilastirmasi (CV Accuracy vs Test Accuracy)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Accuracy (Dogruluk)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, results_df['CV_Mean']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ model_comparison.png")

def plot_precision_recall(model, X_test, y_test):
    """Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        return
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    
    ax.plot(recall_vals, precision_vals, color=COLORS['secondary'], linewidth=3,
            label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color=COLORS['secondary'])
    
    ax.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    ax.set_xlabel('Recall (Duyarlilik)', fontsize=12)
    ax.set_ylabel('Precision (Kesinlik)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ precision_recall_curve.png")

def save_all_metrics(metrics, model_info):
    """Tüm metrikleri CSV olarak kaydet."""
    metrics_df = pd.DataFrame([{
        'Model': model_info['best_model_name'],
        **metrics
    }])
    metrics_df.to_csv('data/processed/best_model_metrics.csv', index=False)
    print("\n📊 Metrikler kaydedildi: data/processed/best_model_metrics.csv")

def main():
    model, model_info, X_train, X_test, y_train, y_test, features = load_model_and_data()
    y_pred, metrics = calculate_metrics(model, X_test, y_test)
    
    print("\n--- Grafikler Oluşturuluyor ---")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)
    plot_feature_importance(model, features)
    plot_model_comparison()
    plot_precision_recall(model, X_test, y_test)
    
    save_all_metrics(metrics, model_info)
    
    print("\n" + "=" * 60)
    print("MODEL DEĞERLENDİRME TAMAMLANDI ✓")
    print("=" * 60)

if __name__ == '__main__':
    main()
