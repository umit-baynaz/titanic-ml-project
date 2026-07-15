# -*- coding: utf-8 -*-
"""
Titanic Veri Analizi - Adım 3: Power BI Hazırlığı
=================================================
Power BI ile etkileşimli dashboard oluşturulması için, 
hayatta kalma oranları, yolcu sınıfları, yaş dağılımı ve cinsiyet 
bazlı analiz verilerini özel rapor tabloları halinde dışa aktarır.
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def main():
    print("=" * 60)
    print("ADIM 3: POWER BI DASHBOARD VERİ HAZIRLIĞI")
    print("=" * 60)
    
    df = pd.read_excel('data/processed/train_cleaned.xlsx')
    os.makedirs('data/powerbi', exist_ok=True)
    
    # 1. Tam Veri Seti (Ana Dashboard Beslemesi)
    powerbi_cols = ['PassengerId', 'Name', 'Sex_Text', 'Age', 'Age_Group', 
                    'Pclass_Text', 'Fare', 'Embarked', 'Survived_Text', 'Survived']
    df_powerbi = df[powerbi_cols].copy()
    df_powerbi.rename(columns={
        'Sex_Text': 'Cinsiyet',
        'Pclass_Text': 'Yolcu Sinifi',
        'Survived_Text': 'Hayatta Kalma Durumu',
        'Age_Group': 'Yas Grubu',
        'Fare': 'Bilet Ucreti',
        'Embarked': 'Binis Limani'
    }, inplace=True)
    
    df_powerbi.to_excel('data/powerbi/01_PowerBI_Ana_Veri.xlsx', index=False, engine='openpyxl')
    print("✅ PowerBI Ana Veri Seti oluşturuldu (Excel formatında).")
    
    # 2. Özet Metrikler - Hayatta Kalma Oranları (Cinsiyet Bazlı)
    gender_survival = df.groupby('Sex_Text')['Survived'].agg(['count', 'sum']).reset_index()
    gender_survival.columns = ['Cinsiyet', 'Toplam Yolcu', 'Hayatta Kalan']
    gender_survival['Hayatta Kalma Orani %'] = (gender_survival['Hayatta Kalan'] / gender_survival['Toplam Yolcu'] * 100).round(2)
    gender_survival.to_excel('data/powerbi/02_PowerBI_Cinsiyet_Analizi.xlsx', index=False, engine='openpyxl')
    
    # 3. Özet Metrikler - Yolcu Sınıfları
    class_survival = df.groupby('Pclass_Text')['Survived'].agg(['count', 'sum']).reset_index()
    class_survival.columns = ['Yolcu Sinifi', 'Toplam Yolcu', 'Hayatta Kalan']
    class_survival['Hayatta Kalma Orani %'] = (class_survival['Hayatta Kalan'] / class_survival['Toplam Yolcu'] * 100).round(2)
    class_survival.to_excel('data/powerbi/03_PowerBI_Sinif_Analizi.xlsx', index=False, engine='openpyxl')
    
    # 4. Özet Metrikler - Yaş Dağılımı
    age_survival = df.groupby('Age_Group')['Survived'].agg(['count', 'sum']).reset_index()
    age_survival.columns = ['Yas Grubu', 'Toplam Yolcu', 'Hayatta Kalan']
    age_survival['Hayatta Kalma Orani %'] = (age_survival['Hayatta Kalan'] / age_survival['Toplam Yolcu'] * 100).round(2)
    age_survival.to_excel('data/powerbi/04_PowerBI_Yas_Dagilimi.xlsx', index=False, engine='openpyxl')
    
    print("✅ Tüm dashboard rapor tabloları 'data/powerbi/' klasörüne aktarıldı.")
    print("\nPower BI'da Dashboard Oluşturmak İçin Kullanılacak Dosyalar:")
    print("1. 01_PowerBI_Ana_Veri.xlsx")
    print("2. 02_PowerBI_Cinsiyet_Analizi.xlsx")
    print("3. 03_PowerBI_Sinif_Analizi.xlsx")
    print("4. 04_PowerBI_Yas_Dagilimi.xlsx")
    
    print("\n" + "=" * 60)
    print("POWER BI HAZIRLIĞI VE TÜM PROJE TAMAMLANDI ✓")
    print("=" * 60)

if __name__ == '__main__':
    main()
