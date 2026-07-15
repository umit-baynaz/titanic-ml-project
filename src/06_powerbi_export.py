# -*- coding: utf-8 -*-
"""
Titanic ML Project - Adım 6: Power BI Veri Export
=================================================
Power BI'da kullanılmak üzere özel formatlı CSV dosyaları oluşturur.
Her CSV bir Power BI rapor sayfasına karşılık gelir.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = 'data/powerbi'

def load_data():
    print("=" * 60)
    print("ADIM 6: POWER BI VERİ EXPORT")
    print("=" * 60)
    
    df_raw = pd.read_csv('data/raw/train.csv')
    df_clean = pd.read_csv('data/processed/train_cleaned.csv')
    df_eng = pd.read_csv('data/processed/train_engineered.csv')
    
    print(f"\n📊 Veriler yüklendi")
    return df_raw, df_clean, df_eng

def export_survival_summary(df):
    """Genel hayatta kalma özeti - KPI kartları için."""
    print("\n🔧 survival_summary.csv oluşturuluyor...")
    
    summary = pd.DataFrame([{
        'Toplam_Yolcu': len(df),
        'Hayatta_Kalan': df['Survived'].sum(),
        'Hayatini_Kaybeden': len(df) - df['Survived'].sum(),
        'Hayatta_Kalma_Orani_Pct': round(df['Survived'].mean() * 100, 2),
        'Ortalama_Yas': round(df['Age'].mean(), 1),
        'Medyan_Yas': round(df['Age'].median(), 1),
        'Ortalama_Ucret': round(df['Fare'].mean(), 2),
        'Medyan_Ucret': round(df['Fare'].median(), 2),
        'Erkek_Sayisi': (df['Sex'] == 'male').sum(),
        'Kadin_Sayisi': (df['Sex'] == 'female').sum(),
        'Sinif1_Sayisi': (df['Pclass'] == 1).sum(),
        'Sinif2_Sayisi': (df['Pclass'] == 2).sum(),
        'Sinif3_Sayisi': (df['Pclass'] == 3).sum()
    }])
    
    summary.to_csv(f'{OUTPUT_DIR}/survival_summary.csv', index=False)
    print("   ✓ Kaydedildi")

def export_class_gender_survival(df):
    """Sınıf ve cinsiyet bazlı hayatta kalma - Çapraz analiz için."""
    print("\n🔧 class_gender_survival.csv oluşturuluyor...")
    
    result = df.groupby(['Pclass', 'Sex']).agg(
        Toplam_Yolcu=('Survived', 'count'),
        Hayatta_Kalan=('Survived', 'sum'),
        Hayatta_Kalma_Orani=('Survived', 'mean'),
        Ortalama_Yas=('Age', 'mean'),
        Ortalama_Ucret=('Fare', 'mean')
    ).reset_index()
    
    result['Hayatta_Kalma_Orani_Pct'] = (result['Hayatta_Kalma_Orani'] * 100).round(2)
    result['Sinif_Adi'] = result['Pclass'].map({1: '1. Sinif (Luks)', 2: '2. Sinif (Orta)', 3: '3. Sinif (Alt)'})
    result['Cinsiyet'] = result['Sex'].map({'male': 'Erkek', 'female': 'Kadin'})
    
    result.to_csv(f'{OUTPUT_DIR}/class_gender_survival.csv', index=False)
    print("   ✓ Kaydedildi")

def export_age_distribution(df):
    """Yaş dağılımı ve grupları - Histogram ve yaş analizi için."""
    print("\n🔧 age_distribution.csv oluşturuluyor...")
    
    # Bireysel yaş verileri
    age_data = df[['Age', 'Survived', 'Sex', 'Pclass']].copy()
    age_data['Hayatta_Kalma'] = age_data['Survived'].map({0: 'Hayatini Kaybeden', 1: 'Hayatta Kalan'})
    age_data['Cinsiyet'] = age_data['Sex'].map({'male': 'Erkek', 'female': 'Kadin'})
    age_data['Sinif_Adi'] = age_data['Pclass'].map({1: '1. Sinif', 2: '2. Sinif', 3: '3. Sinif'})
    
    # Yaş grupları
    age_data['Yas_Grubu'] = pd.cut(age_data['Age'], 
                                     bins=[0, 12, 18, 35, 50, 80],
                                     labels=['Cocuk (0-12)', 'Genc (13-18)', 
                                             'Yetiskin (19-35)', 'Orta Yas (36-50)', 
                                             'Yasli (51+)'])
    
    age_data.to_csv(f'{OUTPUT_DIR}/age_distribution.csv', index=False)
    
    # Yaş grubu özeti
    age_summary = age_data.groupby('Yas_Grubu').agg(
        Toplam=('Survived', 'count'),
        Hayatta_Kalan=('Survived', 'sum'),
        Hayatta_Kalma_Orani_Pct=('Survived', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    age_summary.to_csv(f'{OUTPUT_DIR}/age_group_summary.csv', index=False)
    
    print("   ✓ Kaydedildi")

def export_fare_analysis(df):
    """Bilet ücreti analizi - Finansal dashboard için."""
    print("\n🔧 fare_analysis.csv oluşturuluyor...")
    
    fare_data = df[['Fare', 'Pclass', 'Survived', 'Sex']].copy()
    fare_data['Hayatta_Kalma'] = fare_data['Survived'].map({0: 'Hayatini Kaybeden', 1: 'Hayatta Kalan'})
    fare_data['Sinif_Adi'] = fare_data['Pclass'].map({1: '1. Sinif', 2: '2. Sinif', 3: '3. Sinif'})
    fare_data['Ucret_Grubu'] = pd.qcut(fare_data['Fare'], q=4, 
                                        labels=['Dusuk', 'Orta', 'Yuksek', 'Premium'],
                                        duplicates='drop')
    
    fare_data.to_csv(f'{OUTPUT_DIR}/fare_analysis.csv', index=False)
    
    # Ücret grubu özeti
    fare_summary = fare_data.groupby('Ucret_Grubu').agg(
        Toplam=('Survived', 'count'),
        Hayatta_Kalan=('Survived', 'sum'),
        Ortalama_Ucret=('Fare', 'mean'),
        Hayatta_Kalma_Orani_Pct=('Survived', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    fare_summary.to_csv(f'{OUTPUT_DIR}/fare_group_summary.csv', index=False)
    
    print("   ✓ Kaydedildi")

def export_family_analysis(df):
    """Aile büyüklüğü analizi."""
    print("\n🔧 family_analysis.csv oluşturuluyor...")
    
    family_data = df[['SibSp', 'Parch', 'Survived']].copy()
    family_data['Aile_Buyuklugu'] = family_data['SibSp'] + family_data['Parch'] + 1
    family_data['Yalniz_mi'] = (family_data['Aile_Buyuklugu'] == 1).map({True: 'Evet', False: 'Hayir'})
    family_data['Aile_Kategorisi'] = pd.cut(family_data['Aile_Buyuklugu'],
                                             bins=[0, 1, 3, 5, 12],
                                             labels=['Yalniz', 'Kucuk Aile', 'Orta Aile', 'Buyuk Aile'])
    
    family_summary = family_data.groupby('Aile_Buyuklugu').agg(
        Toplam=('Survived', 'count'),
        Hayatta_Kalan=('Survived', 'sum'),
        Hayatta_Kalma_Orani_Pct=('Survived', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    
    family_summary.to_csv(f'{OUTPUT_DIR}/family_analysis.csv', index=False)
    
    # Kategori bazlı
    cat_summary = family_data.groupby('Aile_Kategorisi').agg(
        Toplam=('Survived', 'count'),
        Hayatta_Kalan=('Survived', 'sum'),
        Hayatta_Kalma_Orani_Pct=('Survived', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    cat_summary.to_csv(f'{OUTPUT_DIR}/family_category_summary.csv', index=False)
    
    print("   ✓ Kaydedildi")

def export_embarked_analysis(df):
    """Biniş limanı analizi."""
    print("\n🔧 embarked_analysis.csv oluşturuluyor...")
    
    port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    
    embarked_data = df.groupby('Embarked').agg(
        Toplam_Yolcu=('Survived', 'count'),
        Hayatta_Kalan=('Survived', 'sum'),
        Hayatta_Kalma_Orani=('Survived', 'mean'),
        Ortalama_Yas=('Age', 'mean'),
        Ortalama_Ucret=('Fare', 'mean')
    ).reset_index()
    
    embarked_data['Liman_Adi'] = embarked_data['Embarked'].map(port_names)
    embarked_data['Hayatta_Kalma_Orani_Pct'] = (embarked_data['Hayatta_Kalma_Orani'] * 100).round(2)
    
    embarked_data.to_csv(f'{OUTPUT_DIR}/embarked_analysis.csv', index=False)
    print("   ✓ Kaydedildi")

def export_model_results():
    """ML model sonuçlarını Power BI formatında export et."""
    print("\n🔧 Model sonuçları export ediliyor...")
    
    # Model karşılaştırma
    if os.path.exists('data/processed/model_comparison.csv'):
        model_comp = pd.read_csv('data/processed/model_comparison.csv')
        model_comp['CV_Mean_Pct'] = (model_comp['CV_Mean'] * 100).round(2)
        model_comp['Test_Accuracy_Pct'] = (model_comp['Test_Accuracy'] * 100).round(2)
        model_comp.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
        print("   ✓ model_comparison.csv")
    
    # Feature importance
    if os.path.exists('data/processed/feature_importance.csv'):
        fi = pd.read_csv('data/processed/feature_importance.csv')
        fi['Importance_Pct'] = (fi['Importance'] * 100).round(2)
        fi.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
        print("   ✓ feature_importance.csv")
    
    # Confusion matrix
    if os.path.exists('data/processed/confusion_matrix_data.csv'):
        import shutil
        shutil.copy('data/processed/confusion_matrix_data.csv', f'{OUTPUT_DIR}/confusion_matrix_data.csv')
        print("   ✓ confusion_matrix_data.csv")
    
    # Best model metrics
    if os.path.exists('data/processed/best_model_metrics.csv'):
        metrics = pd.read_csv('data/processed/best_model_metrics.csv')
        for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
            if col in metrics.columns:
                metrics[f'{col}_Pct'] = (metrics[col] * 100).round(2)
        metrics.to_csv(f'{OUTPUT_DIR}/best_model_metrics.csv', index=False)
        print("   ✓ best_model_metrics.csv")

def export_correlation_matrix(df):
    """Korelasyon matrisini Power BI için export et."""
    print("\n🔧 correlation_matrix.csv oluşturuluyor...")
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(4)
    
    # Long format (Power BI'da matrix visual için ideal)
    corr_long = corr.reset_index().melt(id_vars='index', var_name='Degisken_2', value_name='Korelasyon')
    corr_long.rename(columns={'index': 'Degisken_1'}, inplace=True)
    corr_long.to_csv(f'{OUTPUT_DIR}/correlation_matrix.csv', index=False)
    print("   ✓ Kaydedildi")

def export_full_dataset(df_clean, df_eng):
    """Power BI için birleştirilmiş tam veri seti."""
    print("\n🔧 full_dataset_powerbi.csv oluşturuluyor...")
    
    # Türkçe etiketlerle zenginleştirilmiş veri seti
    export_df = df_clean.copy()
    export_df['Hayatta_Kalma'] = export_df['Survived'].map({0: 'Hayatini Kaybeden', 1: 'Hayatta Kalan'})
    export_df['Cinsiyet'] = export_df['Sex'].map({'male': 'Erkek', 'female': 'Kadin'})
    export_df['Sinif_Adi'] = export_df['Pclass'].map({1: '1. Sinif (Luks)', 2: '2. Sinif (Orta)', 3: '3. Sinif (Alt)'})
    export_df['Liman_Adi'] = export_df['Embarked'].map({'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'})
    export_df['Aile_Buyuklugu'] = export_df['SibSp'] + export_df['Parch'] + 1
    export_df['Yalniz_mi'] = (export_df['Aile_Buyuklugu'] == 1).map({True: 'Evet', False: 'Hayir'})
    export_df['Yas_Grubu'] = pd.cut(export_df['Age'], 
                                      bins=[0, 12, 18, 35, 50, 80],
                                      labels=['Cocuk', 'Genc', 'Yetiskin', 'Orta Yas', 'Yasli'])
    
    # Unvan çıkar
    export_df['Unvan'] = export_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Dr', 'Rev': 'Rev'
    }
    export_df['Unvan'] = export_df['Unvan'].map(title_mapping).fillna('Diger')
    
    export_df.to_csv(f'{OUTPUT_DIR}/full_dataset_powerbi.csv', index=False)
    print("   ✓ Kaydedildi")

def main():
    df_raw, df_clean, df_eng = load_data()
    
    export_survival_summary(df_clean)
    export_class_gender_survival(df_clean)
    export_age_distribution(df_clean)
    export_fare_analysis(df_clean)
    export_family_analysis(df_clean)
    export_embarked_analysis(df_clean)
    export_correlation_matrix(df_clean)
    export_full_dataset(df_clean, df_eng)
    export_model_results()
    
    # Dosya listesi
    print("\n" + "=" * 60)
    print("POWER BI VERİ EXPORT TAMAMLANDI ✓")
    print("=" * 60)
    print(f"\nOluşturulan dosyalar ({OUTPUT_DIR}/):")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.csv'):
            size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024
            print(f"   📄 {f} ({size:.1f} KB)")

if __name__ == '__main__':
    main()
