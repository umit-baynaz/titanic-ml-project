# -*- coding: utf-8 -*-
"""
Titanic ML Project - Adım 4: Model Eğitimi
==========================================
5 farklı ML modeli eğitir, cross-validation yapar ve
en iyi modeli kaydeder.
"""

import pandas as pd
import numpy as np
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def load_data():
    print("=" * 60)
    print("ADIM 4: MODEL EĞİTİMİ")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/ml_features.csv')
    print(f"\n📊 ML veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    print(f"   Özellik sayısı: {X.shape[1]}")
    print(f"   Sınıf dağılımı: {y.value_counts().to_dict()}")
    
    return X, y

def split_and_scale(X, y):
    """Veriyi böl ve ölçeklendir."""
    print("\n🔧 Veri bölünüyor ve ölçeklendiriliyor...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scaler'ı kaydet
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"   Train: {X_train.shape[0]} satır")
    print(f"   Test: {X_test.shape[0]} satır")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()

def define_models():
    """Modelleri tanımla."""
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }
    return models

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Tüm modelleri eğit ve karşılaştır."""
    models = define_models()
    results = []
    best_overall_score = 0
    best_overall_model = None
    best_overall_name = ""
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Model Eğitimi Başlıyor ---")
    print(f"{'Model':<25} {'CV Ort.':<12} {'CV Std':<12} {'Test Acc.':<12}")
    print("-" * 61)
    
    for name, config in models.items():
        print(f"\n🔄 {name} eğitiliyor...")
        
        # GridSearchCV ile hiperparametre optimizasyonu
        grid_search = GridSearchCV(
            config['model'], config['params'],
            cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Cross-validation skorları
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Test seti performansı
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   {name:<25} {cv_scores.mean():.4f}       {cv_scores.std():.4f}       {test_accuracy:.4f}")
        print(f"   En iyi parametreler: {grid_search.best_params_}")
        
        results.append({
            'Model': name,
            'CV_Mean': round(cv_scores.mean(), 4),
            'CV_Std': round(cv_scores.std(), 4),
            'Test_Accuracy': round(test_accuracy, 4),
            'Best_Params': str(grid_search.best_params_)
        })
        
        if cv_scores.mean() > best_overall_score:
            best_overall_score = cv_scores.mean()
            best_overall_model = best_model
            best_overall_name = name
    
    print(f"\n🏆 EN İYİ MODEL: {best_overall_name} (CV: {best_overall_score:.4f})")
    
    return results, best_overall_model, best_overall_name

def save_results(results, best_model, best_name, feature_names):
    """Sonuçları kaydet."""
    # Model karşılaştırma CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('CV_Mean', ascending=False)
    results_df.to_csv('data/processed/model_comparison.csv', index=False)
    print(f"\n📊 Model karşılaştırma: data/processed/model_comparison.csv")
    
    # En iyi modeli kaydet
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"💾 En iyi model kaydedildi: models/best_model.pkl ({best_name})")
    
    # Feature importance (varsa)
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv('data/processed/feature_importance.csv', index=False)
        print(f"📊 Feature importance: data/processed/feature_importance.csv")
        print("\n   En önemli 5 özellik:")
        for _, row in importance_df.head().iterrows():
            print(f"   - {row['Feature']}: {row['Importance']:.4f}")
    
    # Model bilgilerini kaydet
    info = {
        'best_model_name': best_name,
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }
    joblib.dump(info, 'models/model_info.pkl')

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(X, y)
    results, best_model, best_name = train_models(X_train, X_test, y_train, y_test, feature_names)
    save_results(results, best_model, best_name, feature_names)
    
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ TAMAMLANDI ✓")
    print("=" * 60)
    return results, best_model

if __name__ == '__main__':
    main()
