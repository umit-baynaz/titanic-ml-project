# Titanic Veri Analizi ve Görselleştirme Projesi

Bu proje, Titanic yolcu veri seti üzerinde uçtan uca veri analizi süreçlerini (veri temizleme, Keşifsel Veri Analizi - EDA, istatistiksel analiz, görselleştirme ve raporlama) içermektedir.

## 📌 Proje Özeti

- Titanik yolcu verisini **Excel** ortamında inceleyerek veri temizleme ve ön işleme süreçlerini gerçekleştirdim. Titanik veri setini **Python**'da veri görselleştirmesi yapıldı.
- **Python (Pandas ve NumPy)** kullanarak eksik verileri analiz ettim, veri dönüşümleri uyguladım ve istatistiksel analizler gerçekleştirdim.
- **Power BI** kullanarak etkileşimli dashboard oluşturdum; hayatta kalma oranları, yolcu sınıfları, yaş dağılımı ve cinsiyet bazlı analizleri raporladım.
- Veri analizi sürecinde veri temizleme, keşifsel veri analizi (EDA), görselleştirme ve raporlama aşamalarını **uçtan uca yönettim.**

---

## 📂 Proje Dosya Yapısı

```
titanic-ml-project/
├── data/
│   ├── raw/
│   │   └── train.csv                      # Ham Veri Seti
│   ├── processed/
│   │   ├── train_cleaned.xlsx             # Excel ortamında incelemek için temizlenmiş veri
│   │   └── train_cleaned.csv              # Python analizleri için temizlenmiş veri
│   └── powerbi/                           # Power BI Dashboard verileri (Excel formatında)
│       ├── 01_PowerBI_Ana_Veri.xlsx
│       ├── 02_PowerBI_Cinsiyet_Analizi.xlsx
│       ├── 03_PowerBI_Sinif_Analizi.xlsx
│       └── 04_PowerBI_Yas_Dagilimi.xlsx
│
├── reports/
│   └── figures/                           # Python (Matplotlib/Seaborn) Görselleştirmeleri
│       ├── 01_sinif_cinsiyet_analizi.png
│       ├── 02_yas_dagilimi.png
│       └── 03_yas_gruplari_analizi.png
│
├── src/                                   # Python Kaynak Kodları
│   ├── 01_veri_temizleme.py               # Eksik veri analizi ve dönüşümler
│   ├── 02_kesifsel_analiz.py              # İstatistiksel hesaplamalar ve görselleştirme
│   └── 03_powerbi_hazirlik.py             # Raporlama verilerinin oluşturulması
│
└── README.md
```

---

## 🛠️ Kullanılan Araçlar ve Teknolojiler

| Araç | Kullanım Amacı |
|------|----------------|
| **Python** | Veri analizi kodlaması |
| **Pandas & NumPy** | Veri manipülasyonu, eksik veri analizi, istatistikler |
| **Matplotlib & Seaborn** | Keşifsel Veri Analizi (EDA) görselleştirmeleri |
| **Excel** | Veri ön işleme incelemesi ve tablo yapıları |
| **Power BI** | Etkileşimli raporlama (Dashboard) |

---

## 🚀 Analiz Adımları ve Kod Çalıştırma

Projede gerçekleştirilen aşamaları kendi ortamınızda tekrar etmek için terminalinizde aşağıdaki komutları sırasıyla çalıştırabilirsiniz:

### 1. Veri Temizleme ve Ön İşleme
```bash
python src/01_veri_temizleme.py
```
*Bu aşamada Pandas ve NumPy ile eksik veriler tespit edilir, yaş ve liman gibi eksik alanlar istatistiksel yöntemlerle doldurulur ve sonuçlar Excel (.xlsx) ortamına aktarılır.*

### 2. İstatistiksel Analiz ve Görselleştirme (EDA)
```bash
python src/02_kesifsel_analiz.py
```
*Bu aşamada hayatta kalma oranları, cinsiyet, sınıf ve yaş bazlı dağılımlar hesaplanır. Üretilen grafikler `reports/figures/` klasörüne otomatik kaydedilir.*

### 3. Power BI Raporlama Hazırlığı
```bash
python src/03_powerbi_hazirlik.py
```
*Bu aşamada Power BI'da oluşturulacak etkileşimli dashboard için özel olarak filtrelenmiş yaş, cinsiyet, sınıf ve ana veri rapor tabloları Excel formatında dışa aktarılır.*

---

## 📊 Öne Çıkan Bulgular

İstatistiksel analizler sonucunda elde edilen temel bulgular:
1. **Genel Hayatta Kalma:** Gemideki yolcuların yalnızca **%38.4**'ü hayatta kalmıştır.
2. **Cinsiyet Etkisi:** Kadınların hayatta kalma oranı **%74.2** iken, bu oran erkeklerde **%18.9**'da kalmıştır.
3. **Yolcu Sınıfı Etkisi:** 1. sınıf yolcuların hayatta kalma ihtimali (%63.0), 3. sınıf yolculara göre (%24.2) çok daha yüksektir.
4. **Yaş Dağılımı:** Gemideki yolcuların yaş ortalaması 29.1 olarak hesaplanmıştır.
