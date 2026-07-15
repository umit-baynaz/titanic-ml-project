# 🚢 Titanic ML Project - Hayatta Kalma Tahmini

Titanic veri seti üzerinde kapsamlı makine öğrenmesi analizi. Bu proje, 1912 yılında batan RMS Titanic gemisindeki yolcuların hayatta kalma olasılığını tahmin eden bir ML pipeline'ı içerir.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Tamamlandı-green)

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Analiz Sonuçları](#-analiz-sonuçları)
- [ML Modelleri](#-ml-modelleri)
- [Power BI Raporu](#-power-bi-raporu)
- [Görseller](#-görseller)

---

## 🎯 Proje Hakkında

Bu proje, Kaggle Titanic veri seti üzerinde uçtan uca bir veri bilimi pipeline'ı gerçekleştirir:

1. **Veri Temizleme** — Eksik verilerin akıllı doldurulması
2. **Keşifsel Analiz (EDA)** — 8 profesyonel görselleştirme
3. **Feature Engineering** — 15 özellik çıkarımı
4. **Model Eğitimi** — 5 farklı ML algoritması karşılaştırması
5. **Model Değerlendirme** — ROC, Confusion Matrix, Feature Importance
6. **Power BI Export** — Dashboard için hazır CSV dosyaları

### Kullanılan Teknolojiler

| Araç | Kullanım |
|------|----------|
| **Python 3.14** | Veri analizi ve modelleme |
| **pandas & numpy** | Veri işleme |
| **scikit-learn** | ML modelleri |
| **XGBoost** | Gradient Boosting |
| **matplotlib & seaborn** | Görselleştirme |
| **Power BI** | İnteraktif dashboard |

---

## 📁 Proje Yapısı

```
titanic-ml-project/
├── README.md                    # Bu dosya
├── requirements.txt             # Python bağımlılıkları
├── .gitignore                   # Git ignore
│
├── data/
│   ├── raw/                     # Ham veri
│   │   └── train.csv
│   ├── processed/               # İşlenmiş veri
│   │   ├── train_cleaned.csv
│   │   ├── train_engineered.csv
│   │   └── ml_features.csv
│   └── powerbi/                 # Power BI veri dosyaları
│       ├── full_dataset_powerbi.csv
│       ├── survival_summary.csv
│       ├── class_gender_survival.csv
│       ├── model_comparison.csv
│       └── ... (15+ CSV dosyası)
│
├── src/                         # Kaynak kodlar
│   ├── 01_veri_temizleme.py     # Veri temizleme
│   ├── 02_kesifsel_analiz.py    # EDA görselleştirme
│   ├── 03_feature_engineering.py # Özellik mühendisliği
│   ├── 04_model_egitimi.py      # ML model eğitimi
│   ├── 05_model_degerlendirme.py # Model değerlendirme
│   └── 06_powerbi_export.py     # Power BI export
│
├── models/                      # Eğitilmiş modeller
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_info.pkl
│
├── reports/figures/             # Analiz grafikleri (PNG)
│   ├── survival_overview.png
│   ├── age_distribution.png
│   ├── class_gender_survival.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── ...
│
└── powerbi/                     # Power BI kılavuzu
    └── POWERBI_KILAVUZ.md
```

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.10+
- pip

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone https://github.com/KULLANICIADI/titanic-ml-project.git
cd titanic-ml-project

# 2. Bağımlılıkları kurun
pip install -r requirements.txt

# 3. Pipeline'ı çalıştırın (sırasıyla)
python src/01_veri_temizleme.py
python src/02_kesifsel_analiz.py
python src/03_feature_engineering.py
python src/04_model_egitimi.py
python src/05_model_degerlendirme.py
python src/06_powerbi_export.py
```

---

## 🚀 Kullanım

### Tek Komutla Tüm Pipeline

Windows (PowerShell):
```powershell
$env:PYTHONIOENCODING='utf-8'
@("01_veri_temizleme", "02_kesifsel_analiz", "03_feature_engineering", "04_model_egitimi", "05_model_degerlendirme", "06_powerbi_export") | ForEach-Object { python "src/$_.py" }
```

### Adım Adım

| Adım | Script | Açıklama |
|------|--------|----------|
| 1 | `01_veri_temizleme.py` | Eksik veri doldurma, tip düzeltme |
| 2 | `02_kesifsel_analiz.py` | İstatistikler ve 8 grafik |
| 3 | `03_feature_engineering.py` | 15 yeni özellik türetme |
| 4 | `04_model_egitimi.py` | 5 model eğitimi + GridSearch |
| 5 | `05_model_degerlendirme.py` | Metrikler ve performans grafikleri |
| 6 | `06_powerbi_export.py` | Power BI CSV dosyaları |

---

## 📊 Analiz Sonuçları

### Temel Bulgular

| Metrik | Değer |
|--------|-------|
| Toplam Yolcu | 418 |
| Hayatta Kalma Oranı | %36.4 |
| Ortalama Yaş | 29.3 |
| Ortalama Bilet Ücreti | $35.56 |

### Önemli Keşifler

- 🚺 **Kadınlar** erkeklere göre çok daha yüksek hayatta kalma oranına sahip
- 💰 **1. sınıf** yolcuların hayatta kalma şansı 3. sınıftan çok daha fazla
- 👨‍👩‍👧‍👦 **2-4 kişilik aileler** en yüksek hayatta kalma oranına sahip
- 🚢 **Cherbourg** limanından binen yolcuların hayatta kalma oranı en yüksek
- 👶 **Çocuklar** (0-12 yaş) diğer yaş gruplarına göre daha şanslı

---

## 🤖 ML Modelleri

### Karşılaştırılan Modeller

| Model | CV Accuracy | Test Accuracy |
|-------|------------|---------------|
| Logistic Regression | ~%78-82 | ~%78-82 |
| Random Forest | ~%80-84 | ~%79-83 |
| Gradient Boosting | ~%80-84 | ~%79-83 |
| SVM | ~%78-82 | ~%78-82 |
| KNN | ~%75-80 | ~%75-80 |

> **Not:** Yukarıdaki değerler yaklaşıktır. Kesin sonuçlar `data/processed/model_comparison.csv` dosyasında bulunabilir.

### Özellik Mühendisliği (15 Özellik)

```
Pclass, Age, SibSp, Parch, Fare,
Sex_Encoded, Embarked_Encoded, Title_Encoded,
FamilySize, IsAlone, HasCabin, Deck_Encoded,
AgeBand_Encoded, FareBand_Encoded, FamilyCategory_Encoded
```

---

## 📈 Power BI Raporu

Power BI Dashboard 5 sayfadan oluşur:

| Sayfa | İçerik |
|-------|--------|
| **Genel Bakış** | KPI kartları, pasta grafiği, sınıf dağılımı |
| **Demografik Analiz** | Cinsiyet/sınıf/yaş bazlı hayatta kalma |
| **Aile & Sosyal** | Aile büyüklüğü, yalnız yolcu, biniş limanı |
| **Finansal Analiz** | Bilet ücreti dağılımları |
| **ML Sonuçları** | Model karşılaştırma, feature importance |

📖 Detaylı kılavuz: [`powerbi/POWERBI_KILAVUZ.md`](powerbi/POWERBI_KILAVUZ.md)

---

## 🖼️ Görseller

Tüm grafikler `reports/figures/` klasöründe PNG formatında bulunabilir:

- `survival_overview.png` — Genel hayatta kalma analizi
- `age_distribution.png` — Yaş dağılımı
- `class_gender_survival.png` — Sınıf ve cinsiyet çapraz analizi
- `correlation_heatmap.png` — Korelasyon ısı haritası
- `family_analysis.png` — Aile büyüklüğü analizi
- `embarked_analysis.png` — Biniş limanı analizi
- `violin_age.png` — Yaş keman grafiği
- `fare_distribution.png` — Bilet ücreti dağılımı
- `model_comparison.png` — Model performans karşılaştırması
- `confusion_matrix.png` — Karışıklık matrisi
- `roc_curve.png` — ROC eğrisi
- `feature_importance.png` — Özellik önem sıralaması

---

## 📝 Lisans

Bu proje eğitim amaçlıdır. Veri seti [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic) kaynağından alınmıştır.

---

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/yeniOzellik`)
5. Pull Request açın
