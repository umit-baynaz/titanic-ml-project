# -*- coding: utf-8 -*-
"""
Titanic ML Project - Adım 1: Veri Temizleme ve Ön İşleme
=========================================================
Bu script ham Titanic verisini temizler ve analiz için hazırlar.
"""

import pandas as pd
import numpy as np
import os
import sys

# Proje kök dizinini ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def load_data():
    """Ham veriyi yükle."""
    print("=" * 60)
    print("ADIM 1: VERI TEMIZLEME VE ON ISLEME")
    print("=" * 60)
    
    df = pd.read_csv('data/raw/train.csv')
    print(f"\nVeri seti yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
    return df

def analyze_missing(df):
    """Eksik veri analizini yap ve raporla."""
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
        print("Eksik veri bulunmamaktadir.")
    
    return missing_df

def clean_age(df):
    """Yas sutunundaki eksik degerleri Pclass ve Sex gruplarina gore median ile doldur."""
    print("\n[*] Age (Yas) - Grup medyani ile dolduruluyor...")
    before = df['Age'].isnull().sum()
    
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    # Hala eksik varsa genel medyan ile doldur
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    after = df['Age'].isnull().sum()
    print(f"   Doldurulan: {before - after} deger")
    return df

def clean_embarked(df):
    """Embarked sutunundaki eksik degerleri mode ile doldur."""
    if 'Embarked' in df.columns and df['Embarked'].isnull().sum() > 0:
        print("\n[*] Embarked (Liman) - Mode ile dolduruluyor...")
        before = df['Embarked'].isnull().sum()
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        print(f"   Doldurulan: {before} deger")
    return df

def clean_fare(df):
    """Fare sutunundaki eksik degerleri median ile doldur."""
    if 'Fare' in df.columns and df['Fare'].isnull().sum() > 0:
        print("\n[*] Fare (Ucret) - Median ile dolduruluyor...")
        before = df['Fare'].isnull().sum()
        df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)
        print(f"   Doldurulan: {before} deger")
    return df

def clean_cabin(df):
    """Cabin sutununu isle: Deck harfi cikar ve has_cabin flag ekle."""
    print("\n[*] Cabin (Kabin) - Deck cikarma ve has_cabin flag ekleniyor...")
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
    print(f"   HasCabin ve Deck sutunlari eklendi")
    print(f"   Deck dagilimi: {df['Deck'].value_counts().to_dict()}")
    return df

def clean_data(df):
    """Ana temizleme pipeline."""
    df = clean_age(df)
    df = clean_embarked(df)
    df = clean_fare(df)
    df = clean_cabin(df)
    
    # Veri tipi duzeltmeleri
    df['Survived'] = df['Survived'].astype(int)
    df['Pclass'] = df['Pclass'].astype(int)
    
    return df

def save_cleaned(df):
    """Temizlenmis veriyi kaydet."""
    output_path = 'data/processed/train_cleaned.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Temizlenmis veri kaydedildi: {output_path}")
    print(f"   Boyut: {df.shape[0]} satir x {df.shape[1]} sutun")
    print(f"   Eksik veri: {df.isnull().sum().sum()} (toplam)")

def main():
    df = load_data()
    analyze_missing(df)
    df = clean_data(df)
    
    print("\n--- Temizleme Sonrasi Kontrol ---")
    analyze_missing(df)
    
    save_cleaned(df)
    
    print("\n" + "=" * 60)
    print("VERI TEMIZLEME TAMAMLANDI")
    print("=" * 60)
    return df

if __name__ == '__main__':
    main()
