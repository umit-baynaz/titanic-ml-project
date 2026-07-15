# -*- coding: utf-8 -*-
"""
Titanic ML Project - Adim 2: Kesifsel Veri Analizi (EDA)
========================================================
Bu script veri setinin kapsamli istatistiksel analizini yapar
ve gorsellestirmeleri PNG olarak kaydeder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI olmadan calis
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Proje kok dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Renk paleti
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'info': '#3498DB',
    'dark': '#2C3E50',
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#2ECC71', '#9B59B6', '#E74C3C']
}

def load_data():
    print("=" * 60)
    print("ADIM 2: KESIFSEL VERI ANALIZI (EDA)")
    print("=" * 60)
    df = pd.read_csv('data/processed/train_cleaned.csv')
    print(f"\nTemizlenmis veri yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
    return df

def print_statistics(df):
    """Temel istatistikleri yazdir."""
    print("\n--- Temel Istatistikler ---")
    print(f"Toplam Yolcu: {len(df)}")
    print(f"Hayatta Kalan: {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
    print(f"Hayatini Kaybeden: {(1-df['Survived']).sum():.0f} ({(1-df['Survived'].mean())*100:.1f}%)")
    print(f"\nOrtalama Yas: {df['Age'].mean():.1f}")
    print(f"Ortalama Ucret: ${df['Fare'].mean():.2f}")
    print(f"\nCinsiyet Dagilimi:")
    for sex, count in df['Sex'].value_counts().items():
        print(f"  {sex}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nSinif Dagilimi:")
    for pclass, count in df['Pclass'].value_counts().sort_index().items():
        print(f"  {pclass}. Sinif: {count} ({count/len(df)*100:.1f}%)")

def plot_survival_overview(df):
    """Genel hayatta kalma dagilimi."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Titanic - Hayatta Kalma Genel Bakis', fontsize=16, fontweight='bold', y=1.02)
    
    # Pasta grafigi
    survived_counts = df['Survived'].value_counts()
    labels = ['Hayatini Kaybeden', 'Hayatta Kalan']
    colors_pie = [COLORS['danger'], COLORS['success']]
    axes[0].pie(survived_counts.values, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Hayatta Kalma Orani', fontsize=14)
    
    # Sinifa gore
    survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
    bars = axes[1].bar(survival_by_class.index, survival_by_class.values,
                       color=COLORS['palette'][:3], edgecolor='white', linewidth=1.5)
    axes[1].set_title('Sinifa Gore Hayatta Kalma (%)', fontsize=14)
    axes[1].set_xlabel('Yolcu Sinifi')
    axes[1].set_ylabel('Hayatta Kalma Orani (%)')
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['1. Sinif (Luks)', '2. Sinif (Orta)', '3. Sinif (Alt)'])
    for bar, val in zip(bars, survival_by_class.values):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Cinsiyete gore
    survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
    colors_sex = [COLORS['secondary'], COLORS['primary']]
    bars = axes[2].bar(survival_by_sex.index, survival_by_sex.values,
                       color=colors_sex, edgecolor='white', linewidth=1.5)
    axes[2].set_title('Cinsiyete Gore Hayatta Kalma (%)', fontsize=14)
    axes[2].set_xlabel('Cinsiyet')
    axes[2].set_ylabel('Hayatta Kalma Orani (%)')
    for bar, val in zip(bars, survival_by_sex.values):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/figures/survival_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > survival_overview.png")

def plot_age_distribution(df):
    """Yas dagilimi analizi."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Yas Dagilimi Analizi', fontsize=16, fontweight='bold', y=1.02)
    
    for survived, color, label in [(0, COLORS['danger'], 'Hayatini Kaybeden'),
                                    (1, COLORS['success'], 'Hayatta Kalan')]:
        subset = df[df['Survived'] == survived]['Age']
        axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    axes[0].set_title('Yas Dagilimi (Hayatta Kalma Durumuna Gore)', fontsize=14)
    axes[0].set_xlabel('Yas')
    axes[0].set_ylabel('Yolcu Sayisi')
    axes[0].legend()
    
    sns.boxplot(data=df, x='Pclass', y='Age', hue='Survived',
                palette={0: COLORS['danger'], 1: COLORS['success']}, ax=axes[1])
    axes[1].set_title('Sinif ve Hayatta Kalma Durumuna Gore Yas', fontsize=14)
    axes[1].set_xlabel('Yolcu Sinifi')
    axes[1].set_ylabel('Yas')
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, ['Hayatini Kaybeden', 'Hayatta Kalan'])
    
    plt.tight_layout()
    plt.savefig('reports/figures/age_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > age_distribution.png")

def plot_fare_distribution(df):
    """Bilet ucreti analizi."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Bilet Ucreti Analizi', fontsize=16, fontweight='bold', y=1.02)
    
    sns.histplot(data=df, x='Fare', hue='Survived', bins=40,
                 palette={0: COLORS['danger'], 1: COLORS['success']},
                 alpha=0.6, ax=axes[0])
    axes[0].set_title('Ucret Dagilimi', fontsize=14)
    axes[0].set_xlabel('Bilet Ucreti ($)')
    axes[0].set_ylabel('Yolcu Sayisi')
    
    sns.boxplot(data=df, x='Pclass', y='Fare',
                palette=COLORS['palette'][:3], ax=axes[1])
    axes[1].set_title('Sinifa Gore Ucret Dagilimi', fontsize=14)
    axes[1].set_xlabel('Yolcu Sinifi')
    axes[1].set_ylabel('Bilet Ucreti ($)')
    
    plt.tight_layout()
    plt.savefig('reports/figures/fare_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > fare_distribution.png")

def plot_class_gender_survival(df):
    """Sinif ve cinsiyet bazli hayatta kalma."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    pivot = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack() * 100
    pivot.plot(kind='bar', ax=ax, color=[COLORS['secondary'], COLORS['primary']],
               edgecolor='white', linewidth=1.5, width=0.7)
    
    ax.set_title('Sinif ve Cinsiyete Gore Hayatta Kalma Orani', fontsize=16, fontweight='bold')
    ax.set_xlabel('Yolcu Sinifi', fontsize=12)
    ax.set_ylabel('Hayatta Kalma Orani (%)', fontsize=12)
    ax.set_xticklabels(['1. Sinif', '2. Sinif', '3. Sinif'], rotation=0)
    ax.legend(['Kadin', 'Erkek'], fontsize=11)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/class_gender_survival.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > class_gender_survival.png")

def plot_embarked_analysis(df):
    """Binis limani analizi."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Binis Limani Analizi', fontsize=16, fontweight='bold', y=1.02)
    
    port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    df_temp = df.copy()
    df_temp['Port'] = df_temp['Embarked'].map(port_names)
    
    port_counts = df_temp['Port'].value_counts()
    axes[0].bar(port_counts.index, port_counts.values, color=COLORS['palette'][:3],
                edgecolor='white', linewidth=1.5)
    axes[0].set_title('Limana Gore Yolcu Sayisi', fontsize=14)
    axes[0].set_xlabel('Binis Limani')
    axes[0].set_ylabel('Yolcu Sayisi')
    
    survival_by_port = df_temp.groupby('Port')['Survived'].mean() * 100
    bars = axes[1].bar(survival_by_port.index, survival_by_port.values,
                       color=COLORS['palette'][:3], edgecolor='white', linewidth=1.5)
    axes[1].set_title('Limana Gore Hayatta Kalma (%)', fontsize=14)
    axes[1].set_xlabel('Binis Limani')
    axes[1].set_ylabel('Hayatta Kalma Orani (%)')
    for bar, val in zip(bars, survival_by_port.values):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/figures/embarked_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > embarked_analysis.png")

def plot_family_analysis(df):
    """Aile buyuklugu analizi."""
    df_temp = df.copy()
    df_temp['FamilySize'] = df_temp['SibSp'] + df_temp['Parch'] + 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Aile Buyuklugu Analizi', fontsize=16, fontweight='bold', y=1.02)
    
    family_counts = df_temp['FamilySize'].value_counts().sort_index()
    axes[0].bar(family_counts.index, family_counts.values, color=COLORS['primary'],
                edgecolor='white', linewidth=1.5)
    axes[0].set_title('Aile Buyuklugu Dagilimi', fontsize=14)
    axes[0].set_xlabel('Ailedeki Kisi Sayisi')
    axes[0].set_ylabel('Yolcu Sayisi')
    
    family_survival = df_temp.groupby('FamilySize')['Survived'].mean() * 100
    axes[1].plot(family_survival.index, family_survival.values, 'o-',
                 color=COLORS['secondary'], linewidth=2.5, markersize=10, markeredgecolor='white')
    axes[1].fill_between(family_survival.index, family_survival.values, alpha=0.15, color=COLORS['secondary'])
    axes[1].set_title('Aile Buyuklugu vs Hayatta Kalma Orani', fontsize=14)
    axes[1].set_xlabel('Ailedeki Kisi Sayisi')
    axes[1].set_ylabel('Hayatta Kalma Orani (%)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/family_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > family_analysis.png")

def plot_correlation_heatmap(df):
    """Korelasyon matrisi isi haritasi."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Korelasyon Katsayisi'})
    
    ax.set_title('Degiskenler Arasi Korelasyon Matrisi', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > correlation_heatmap.png")

def plot_violin_age(df):
    """Keman grafigi - Yas dagilimi."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    sns.violinplot(data=df, x='Pclass', y='Age', hue='Survived',
                   split=True, inner='quart',
                   palette={0: COLORS['danger'], 1: COLORS['success']}, ax=ax)
    
    ax.set_title('Sinif ve Hayatta Kalma Durumuna Gore Yas Dagilimi (Keman Grafigi)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Yolcu Sinifi', fontsize=12)
    ax.set_ylabel('Yas', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Hayatini Kaybeden', 'Hayatta Kalan'], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/violin_age.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  > violin_age.png")

def main():
    df = load_data()
    print_statistics(df)
    
    print("\n--- Grafikler Olusturuluyor ---")
    plot_survival_overview(df)
    plot_age_distribution(df)
    plot_fare_distribution(df)
    plot_class_gender_survival(df)
    plot_embarked_analysis(df)
    plot_family_analysis(df)
    plot_correlation_heatmap(df)
    plot_violin_age(df)
    
    print("\n" + "=" * 60)
    print("KESIFSEL ANALIZ TAMAMLANDI")
    print(f"Grafikler: reports/figures/ klasorune kaydedildi")
    print("=" * 60)

if __name__ == '__main__':
    main()
