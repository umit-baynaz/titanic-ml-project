# 🎯 Titanic Power BI Rapor Kılavuzu

Bu kılavuz, Python scriptleri tarafından oluşturulan CSV dosyalarını Power BI Desktop'ta nasıl kullanacağınızı adım adım anlatır.

---

## 📁 Veri Dosyaları (data/powerbi/)

| Dosya | Kullanım Alanı | Power BI Görsel Önerisi |
|-------|----------------|-------------------------|
| `survival_summary.csv` | KPI Kartları | Card, Multi-row Card |
| `full_dataset_powerbi.csv` | Ana veri seti (tüm analizler) | Tüm görseller |
| `class_gender_survival.csv` | Çapraz analiz | Clustered Bar Chart |
| `age_distribution.csv` | Yaş analizi | Histogram, Scatter |
| `age_group_summary.csv` | Yaş grubu özeti | Bar Chart |
| `fare_analysis.csv` | Ücret analizi | Box Plot, Scatter |
| `fare_group_summary.csv` | Ücret grubu özeti | Donut Chart |
| `family_analysis.csv` | Aile büyüklüğü | Line + Clustered Bar |
| `family_category_summary.csv` | Aile kategorisi | Pie Chart |
| `embarked_analysis.csv` | Liman analizi | Map, Bar Chart |
| `model_comparison.csv` | ML model sonuçları | Clustered Bar |
| `feature_importance.csv` | Özellik önemi | Horizontal Bar |
| `confusion_matrix_data.csv` | Confusion matrix | Matrix, Table |
| `best_model_metrics.csv` | En iyi model metrikleri | Card |
| `correlation_matrix.csv` | Korelasyon matrisi | Matrix Heatmap |

---

## 📋 Adım Adım Rapor Oluşturma

### Adım 1: Power BI Desktop'ı Açın
1. Power BI Desktop'ı başlatın
2. **Get Data** > **Text/CSV** seçin

### Adım 2: Veri İçeri Aktarma
1. `data/powerbi/` klasöründeki CSV dosyalarını tek tek import edin
2. Her dosya için **Load** butonuna tıklayın
3. Alternatif olarak tümünü bir seferde: **Get Data** > **Folder** > `data/powerbi/` seçin

### Adım 3: Veri Modelini Düzenleyin
1. **Model** sekmesine gidin
2. İlişkileri kontrol edin (genellikle otomatik algılanır)

---

## 📊 Sayfa 1: Genel Bakış (Dashboard)

**Kullanılacak Veriler:** `survival_summary.csv`, `full_dataset_powerbi.csv`

### KPI Kartları (Üst Kısım)
Dört adet **Card** görseli ekleyin:

| Kart | Alan | Format |
|------|------|--------|
| Toplam Yolcu | `Toplam_Yolcu` | Tam sayı |
| Hayatta Kalma Oranı | `Hayatta_Kalma_Orani_Pct` | Yüzde |
| Ortalama Yaş | `Ortalama_Yas` | 1 ondalık |
| Ortalama Ücret | `Ortalama_Ucret` | Para ($) |

### Pasta Grafiği (Sol)
- Görsel: **Donut Chart**
- Legend: `Hayatta_Kalma`
- Values: Count of `PassengerId`
- Renkler: Yeşil (Hayatta Kalan) / Kırmızı (Hayatını Kaybeden)

### Sınıf Dağılımı (Sağ)
- Görsel: **Stacked Bar Chart**
- Axis: `Sinif_Adi`
- Values: Count of `PassengerId`
- Legend: `Hayatta_Kalma`

---

## 📊 Sayfa 2: Demografik Analiz

**Kullanılacak Veriler:** `full_dataset_powerbi.csv`, `age_group_summary.csv`, `class_gender_survival.csv`

### Sınıf ve Cinsiyete Göre Hayatta Kalma
- Görsel: **Clustered Bar Chart**
- Veri: `class_gender_survival.csv`
- Axis: `Sinif_Adi`
- Values: `Hayatta_Kalma_Orani_Pct`
- Legend: `Cinsiyet`

### Yaş Dağılımı
- Görsel: **Histogram** (veya **Area Chart**)
- Veri: `full_dataset_powerbi.csv`
- Axis: `Age` (bins oluşturun)
- Values: Count
- Legend: `Hayatta_Kalma`

### Yaş Grupları Özeti
- Görsel: **Bar Chart** + **Line**
- Veri: `age_group_summary.csv`
- Axis: `Yas_Grubu`
- Column values: `Toplam`
- Line values: `Hayatta_Kalma_Orani_Pct`

---

## 📊 Sayfa 3: Aile ve Sosyal Analiz

**Kullanılacak Veriler:** `family_analysis.csv`, `family_category_summary.csv`, `embarked_analysis.csv`

### Aile Büyüklüğü vs Hayatta Kalma
- Görsel: **Line and Clustered Column Chart**
- Veri: `family_analysis.csv`
- Axis: `Aile_Buyuklugu`
- Column: `Toplam`
- Line: `Hayatta_Kalma_Orani_Pct`

### Aile Kategorisi
- Görsel: **Donut Chart**
- Veri: `family_category_summary.csv`
- Legend: `Aile_Kategorisi`
- Values: `Toplam`

### Biniş Limanı
- Görsel: **Clustered Bar Chart**
- Veri: `embarked_analysis.csv`
- Axis: `Liman_Adi`
- Values: `Toplam_Yolcu`, `Hayatta_Kalma_Orani_Pct`

---

## 📊 Sayfa 4: Finansal Analiz

**Kullanılacak Veriler:** `fare_analysis.csv`, `fare_group_summary.csv`

### Ücret Dağılımı
- Görsel: **Scatter Plot**
- Veri: `fare_analysis.csv`
- X axis: `Fare`
- Y axis: (satır numarası veya Age)
- Legend: `Hayatta_Kalma`

### Ücret Grupları Özeti
- Görsel: **Donut Chart** + **Table**
- Veri: `fare_group_summary.csv`
- Legend: `Ucret_Grubu`
- Values: `Hayatta_Kalma_Orani_Pct`

---

## 📊 Sayfa 5: ML Model Sonuçları

**Kullanılacak Veriler:** `model_comparison.csv`, `feature_importance.csv`, `best_model_metrics.csv`

### Model Karşılaştırma
- Görsel: **Clustered Bar Chart**
- Veri: `model_comparison.csv`
- Axis: `Model`
- Values: `CV_Mean_Pct`, `Test_Accuracy_Pct`

### En İyi Model Metrikleri (KPI Kartları)
- Görsel: 4 adet **Card**
- Veri: `best_model_metrics.csv`
- Accuracy, Precision, Recall, F1_Score

### Feature Importance
- Görsel: **Bar Chart** (Horizontal)
- Veri: `feature_importance.csv`
- Y Axis: `Feature`
- X Axis: `Importance_Pct`
- Sıralama: Büyükten küçüğe

---

## 🎨 Tasarım Önerileri

### Renk Paleti
- **Ana Renk:** #2E86AB (Mavi)
- **İkincil:** #A23B72 (Mor)
- **Başarı/Hayatta Kalan:** #2ECC71 (Yeşil)
- **Tehlike/Ölen:** #E74C3C (Kırmızı)
- **Uyarı:** #F39C12 (Turuncu)
- **Arka Plan:** #F8F9FA (Açık Gri)

### Genel İpuçları
1. Her sayfada tutarlı renk kullanın
2. KPI kartlarına koşullu biçimlendirme ekleyin
3. Slicer (filtre) ekleyin: Sınıf, Cinsiyet, Liman
4. Tooltip'leri özelleştirin
5. Sayfa navigasyonu için düğmeler ekleyin

---

## 📐 DAX Formülleri (Opsiyonel Ölçüler)

```dax
// Hayatta Kalma Oranı
Hayatta Kalma Oranı = 
DIVIDE(
    COUNTROWS(FILTER(full_dataset_powerbi, full_dataset_powerbi[Survived] = 1)),
    COUNTROWS(full_dataset_powerbi),
    0
) * 100

// Ortalama Yaş
Ort Yaş = AVERAGE(full_dataset_powerbi[Age])

// Ortalama Ücret
Ort Ücret = AVERAGE(full_dataset_powerbi[Fare])

// Yolcu Sayısı
Yolcu Sayısı = COUNTROWS(full_dataset_powerbi)

// Kadın Hayatta Kalma Oranı
Kadın Hayatta Kalma = 
DIVIDE(
    CALCULATE(COUNTROWS(full_dataset_powerbi), 
              full_dataset_powerbi[Sex] = "female",
              full_dataset_powerbi[Survived] = 1),
    CALCULATE(COUNTROWS(full_dataset_powerbi), 
              full_dataset_powerbi[Sex] = "female"),
    0
) * 100
```

---

## ✅ Kontrol Listesi

- [ ] Tüm CSV dosyaları import edildi
- [ ] Sayfa 1: Genel Bakış oluşturuldu
- [ ] Sayfa 2: Demografik Analiz oluşturuldu
- [ ] Sayfa 3: Aile & Sosyal oluşturuldu
- [ ] Sayfa 4: Finansal Analiz oluşturuldu
- [ ] Sayfa 5: ML Sonuçları oluşturuldu
- [ ] Slicer'lar eklendi
- [ ] Renkler tutarlı
- [ ] .pbix dosyası kaydedildi
