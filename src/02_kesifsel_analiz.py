# -*- coding: utf-8 -*-
"""
Titanic Veri Analizi - Adım 2: Keşifsel Veri Analizi (EDA) ve İstatistik
========================================================================
Python ile istatistiksel analizler gerçekleştirir ve görselleştirmeler yapar.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#2ECC71', 'danger': '#E74C3C'}

def load_data():
    print("=" * 60)
    print("ADIM 2: İSTATİSTİKSEL ANALİZLER VE GÖRSELLEŞTİRME")
    print("=" * 60)
    return pd.read_excel('data/processed/train_cleaned.xlsx')

def perform_statistical_analysis(df):
    """Python (Pandas ve NumPy) kullanarak istatistiksel analizler."""
    print("\n--- İstatistiksel Analiz Sonuçları ---")
    
    # Hayatta Kalma Oranları (Genel)
    survival_rate = df['Survived'].mean() * 100
    print(f"1. Genel Hayatta Kalma Oranı: %{survival_rate:.1f}")
    
    # Cinsiyet Bazlı Analiz (NumPy ve Pandas ile)
    print("\n2. Cinsiyet Bazlı Hayatta Kalma Analizi:")
    gender_stats = df.groupby('Sex_Text')['Survived'].agg(['count', 'mean'])
    gender_stats['mean'] = (gender_stats['mean'] * 100).round(1)
    print(gender_stats.rename(columns={'count': 'Toplam Kişi', 'mean': 'Hayatta Kalma (%)'}))
    
    # Yolcu Sınıfı Analizi
    print("\n3. Yolcu Sınıflarına Göre Hayatta Kalma Analizi:")
    class_stats = df.groupby('Pclass_Text')['Survived'].agg(['count', 'mean'])
    class_stats['mean'] = (class_stats['mean'] * 100).round(1)
    print(class_stats.rename(columns={'count': 'Toplam Kişi', 'mean': 'Hayatta Kalma (%)'}))
    
    # Yaş Dağılımı İstatistikleri
    print("\n4. Yaş Dağılımı ve Temel İstatistikler:")
    print(f"   - Ortalama Yaş: {np.mean(df['Age']):.1f}")
    print(f"   - Medyan Yaş: {np.median(df['Age']):.1f}")
    print(f"   - En Genç: {np.min(df['Age']):.1f} | En Yaşlı: {np.max(df['Age']):.1f}")

def plot_visualizations(df):
    """Veri görselleştirmeleri (EDA)."""
    os.makedirs('reports/figures', exist_ok=True)
    print("\n--- Görselleştirme Raporları Oluşturuluyor ---")
    
    # 1. Cinsiyet ve Sınıf Kırılımında Hayatta Kalma
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Pclass_Text', y='Survived', hue='Sex_Text', ci=None, 
                palette={'Erkek': COLORS['primary'], 'Kadın': COLORS['secondary']})
    plt.title('Sınıf ve Cinsiyete Göre Hayatta Kalma Oranları', fontsize=14, fontweight='bold')
    plt.ylabel('Hayatta Kalma Oranı')
    plt.xlabel('Yolcu Sınıfı')
    plt.savefig('reports/figures/01_sinif_cinsiyet_analizi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Yaş Dağılımı ve Hayatta Kalma
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived_Text', multiple="stack", bins=30,
                 palette={'Hayatını Kaybeden': COLORS['danger'], 'Hayatta Kalan': COLORS['success']})
    plt.title('Yaş Dağılımına Göre Hayatta Kalma Durumu', fontsize=14, fontweight='bold')
    plt.xlabel('Yaş')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('reports/figures/02_yas_dagilimi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Yaş Grupları Analizi
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Age_Group', hue='Survived_Text',
                  palette={'Hayatını Kaybeden': COLORS['danger'], 'Hayatta Kalan': COLORS['success']})
    plt.title('Yaş Gruplarına Göre Hayatta Kalma Analizi', fontsize=14, fontweight='bold')
    plt.xlabel('Yaş Grubu')
    plt.ylabel('Kişi Sayısı')
    plt.savefig('reports/figures/03_yas_gruplari_analizi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Tüm Python veri görselleştirmeleri 'reports/figures/' klasörüne kaydedildi.")

def main():
    df = load_data()
    perform_statistical_analysis(df)
    plot_visualizations(df)
    
    print("\n" + "=" * 60)
    print("İSTATİSTİKSEL ANALİZ VE GÖRSELLEŞTİRME TAMAMLANDI ✓")
    print("=" * 60)

if __name__ == '__main__':
    main()
