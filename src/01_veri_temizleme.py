# -*- coding: utf-8 -*-
"""
Titanic Veri Analizi - Adım 1: Veri Temizleme ve Ön İşleme
==========================================================
Bu script ham Titanic verisini Pandas ve NumPy ile temizler,
eksik verileri analiz eder ve Excel/CSV ortamına aktarır.
"""

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def load_data():
    print("=" * 60)
    print("ADIM 1: VERİ TEMİZLEME VE ÖN İŞLEME")
    print("=" * 60)
    df = pd.read_csv('data/raw/train.csv')
    print(f"\n📊 Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df

def analyze_missing(df):
    """Python (Pandas ve NumPy) kullanarak eksik verileri analiz etme."""
    print("\n--- Eksik Veri Analizi ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Eksik_Sayisi': missing,
        'Eksik_Yuzdesi': missing_pct
    })
    missing_df = missing_df[missing_df['Eksik_Sayisi'] > 0].sort_values('Eksik_Yuzdesi', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("Eksik veri bulunmamaktadır.")
    
    return missing_df

def apply_transformations(df):
    """Veri dönüşümleri ve istatistiksel düzeltmeler."""
    print("\n--- Veri Dönüşümleri Uygulanıyor ---")
    
    # 1. Yaş (Age) eksiklerini Pclass ve Sex grubunun medyanı ile doldurma
    before_age = df['Age'].isnull().sum()
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median()) # Kalanlar için genel medyan
    print(f"🔧 'Age' sütununda {before_age} eksik veri grup medyanı ile dolduruldu.")
    
    # 2. Embarked eksiklerini mod (en çok tekrar eden) ile doldurma
    before_embarked = df['Embarked'].isnull().sum()
    if before_embarked > 0:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        print(f"🔧 'Embarked' sütununda {before_embarked} eksik veri mod ile dolduruldu.")
        
    # 3. Fare eksiklerini medyan ile doldurma
    before_fare = df['Fare'].isnull().sum()
    if before_fare > 0:
        df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
        print(f"🔧 'Fare' sütununda {before_fare} eksik veri medyan ile dolduruldu.")
        
    # 4. Kategorik Dönüşümler (Analizleri kolaylaştırmak için okunabilir metinler)
    df['Survived_Text'] = df['Survived'].map({0: 'Hayatını Kaybeden', 1: 'Hayatta Kalan'})
    df['Pclass_Text'] = df['Pclass'].map({1: '1. Sınıf', 2: '2. Sınıf', 3: '3. Sınıf'})
    df['Sex_Text'] = df['Sex'].map({'male': 'Erkek', 'female': 'Kadın'})
    
    # 5. Yaş Grupları Oluşturma
    df['Age_Group'] = pd.cut(df['Age'], 
                             bins=[0, 12, 18, 35, 50, 100], 
                             labels=['Çocuk', 'Genç', 'Yetişkin', 'Orta Yaş', 'Yaşlı'])
    
    print("✅ Veri dönüşümleri başarıyla tamamlandı.")
    return df

def save_data(df):
    """Excel ortamında inceleme için dışa aktarma."""
    os.makedirs('data/processed', exist_ok=True)
    
    # CSV Formatı
    csv_path = 'data/processed/train_cleaned.csv'
    df.to_csv(csv_path, index=False)
    
    # Excel Formatı (Excel ortamında inceleme için)
    excel_path = 'data/processed/train_cleaned.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    print(f"\n💾 Veriler analiz için kaydedildi:")
    print(f"   - CSV Formatı: {csv_path}")
    print(f"   - Excel Formatı: {excel_path} (Excel ortamında incelemek için)")

def main():
    df = load_data()
    analyze_missing(df)
    df = apply_transformations(df)
    
    print("\n--- Temizleme Sonrası Kontrol ---")
    analyze_missing(df)
    
    save_data(df)
    
    print("\n" + "=" * 60)
    print("VERİ TEMİZLEME VE ÖN İŞLEME AŞAMASI TAMAMLANDI ✓")
    print("=" * 60)

if __name__ == '__main__':
    main()
