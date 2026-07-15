# -*- coding: utf-8 -*-
"""
Titanic ML Project - Adim 3: Ozellik Muhendisligi (Feature Engineering)
======================================================================
Makine ogrenmesi icin yeni ozellikler turetir ve veriyi encode eder.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Proje kok dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def load_data():
    print("=" * 60)
    print("ADIM 3: OZELLIK MUHENDISLIGI (FEATURE ENGINEERING)")
    print("=" * 60)
    df = pd.read_csv('data/processed/train_cleaned.csv')
    print(f"\nTemizlenmis veri yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
    return df

def extract_title(df):
    """Isimden unvan cikar."""
    print("\n[*] Title (Unvan) cikariliyor...")
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Rare',
        'Countess': 'Rare', 'Capt': 'Rare', 'Sir': 'Rare', 'Don': 'Rare',
        'Dona': 'Rare', 'Jonkheer': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    
    print(f"   Unvan dagilimi: {df['Title'].value_counts().to_dict()}")
    return df

def create_family_features(df):
    """Aile ile ilgili ozellikler olustur."""
    print("\n[*] Aile ozellikleri olusturuluyor...")
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    df['FamilyCategory'] = pd.cut(df['FamilySize'],
                                   bins=[0, 1, 3, 5, 12],
                                   labels=['Yalniz', 'Kucuk', 'Orta', 'Buyuk'])
    
    print(f"   FamilySize araligi: {df['FamilySize'].min()} - {df['FamilySize'].max()}")
    print(f"   Yalniz yolcu orani: {df['IsAlone'].mean()*100:.1f}%")
    return df

def create_age_bands(df):
    """Yas gruplari olustur."""
    print("\n[*] Yas gruplari olusturuluyor...")
    
    df['AgeBand'] = pd.cut(df['Age'],
                           bins=[0, 12, 18, 35, 50, 80],
                           labels=['Cocuk', 'Genc', 'Yetiskin', 'OrtaYas', 'Yasli'])
    
    print(f"   Yas grubu dagilimi: {df['AgeBand'].value_counts().to_dict()}")
    return df

def create_fare_bands(df):
    """Ucret gruplari olustur."""
    print("\n[*] Ucret gruplari olusturuluyor...")
    
    df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=['Dusuk', 'Orta', 'Yuksek', 'Premium'])
    
    print(f"   Ucret grubu dagilimi: {df['FareBand'].value_counts().to_dict()}")
    return df

def create_ticket_features(df):
    """Bilet numarasindan ozellikler cikar."""
    print("\n[*] Bilet ozellikleri olusturuluyor...")
    
    df['TicketPrefix'] = df['Ticket'].apply(
        lambda x: x.split()[0] if not x.isdigit() and len(x.split()) > 1 else 'NONE'
    )
    df['TicketLen'] = df['Ticket'].apply(len)
    
    return df

def encode_features(df):
    """Kategorik degiskenleri encode et."""
    print("\n[*] Kategorik degiskenler encode ediliyor...")
    
    label_maps = {
        'Sex': {'male': 0, 'female': 1},
        'Embarked': {'S': 0, 'C': 1, 'Q': 2},
        'Title': {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4},
        'Deck': {'U': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    }
    
    for col, mapping in label_maps.items():
        if col in df.columns:
            df[f'{col}_Encoded'] = df[col].map(mapping).fillna(0).astype(int)
            print(f"   {col} -> {col}_Encoded")
    
    age_band_map = {'Cocuk': 0, 'Genc': 1, 'Yetiskin': 2, 'OrtaYas': 3, 'Yasli': 4}
    fare_band_map = {'Dusuk': 0, 'Orta': 1, 'Yuksek': 2, 'Premium': 3}
    family_cat_map = {'Yalniz': 0, 'Kucuk': 1, 'Orta': 2, 'Buyuk': 3}
    
    df['AgeBand_Encoded'] = df['AgeBand'].map(age_band_map).fillna(0).astype(int)
    df['FareBand_Encoded'] = df['FareBand'].map(fare_band_map).fillna(0).astype(int)
    df['FamilyCategory_Encoded'] = df['FamilyCategory'].map(family_cat_map).fillna(0).astype(int)
    
    return df

def select_ml_features(df):
    """ML icin kullanilacak ozellikleri sec."""
    print("\n[*] ML icin ozellikler seciliyor...")
    
    ml_features = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
        'Sex_Encoded', 'Embarked_Encoded', 'Title_Encoded',
        'FamilySize', 'IsAlone', 'HasCabin', 'Deck_Encoded',
        'AgeBand_Encoded', 'FareBand_Encoded', 'FamilyCategory_Encoded'
    ]
    
    available = [f for f in ml_features if f in df.columns]
    print(f"   Secilen {len(available)} ozellik: {available}")
    
    return df, available

def save_engineered(df, features):
    """Muhendislik yapilmis veriyi kaydet."""
    output_path = 'data/processed/train_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Muhendislik yapilmis veri kaydedildi: {output_path}")
    print(f"   Boyut: {df.shape[0]} satir x {df.shape[1]} sutun")
    
    features_path = 'data/processed/ml_features.csv'
    ml_df = df[['Survived'] + features]
    ml_df.to_csv(features_path, index=False)
    print(f"   ML veri seti: {features_path} ({len(features)} ozellik)")

def main():
    df = load_data()
    
    df = extract_title(df)
    df = create_family_features(df)
    df = create_age_bands(df)
    df = create_fare_bands(df)
    df = create_ticket_features(df)
    df = encode_features(df)
    df, features = select_ml_features(df)
    
    save_engineered(df, features)
    
    print("\n" + "=" * 60)
    print("OZELLIK MUHENDISLIGI TAMAMLANDI")
    print("=" * 60)
    return df, features

if __name__ == '__main__':
    main()
