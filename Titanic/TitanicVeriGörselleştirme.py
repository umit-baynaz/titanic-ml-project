import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. TEMİZLİK: Önceki pencereleri kapat
plt.close('all')

# 2. VERİYİ YÜKLE (Senin dosya ismin 'titanic.csv')
df = pd.read_csv('titanic.csv')

# 3. TÜRKÇELEŞTİRME (Sunum Kalitesi İçin)
try:
    dict_survived = {0: 'Ölü', 1: 'Sağ'}
    dict_pclass = {1: '1. Sınıf (Lüks)', 2: '2. Sınıf (Orta)', 3: '3. Sınıf (Alt)'}
    dict_sex = {'male': 'Erkek', 'female': 'Kadın'}
    dict_embarked = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}

    df['Survived'] = df['Survived'].map(dict_survived)
    df['Pclass'] = df['Pclass'].map(dict_pclass)
    df['Sex'] = df['Sex'].map(dict_sex)
    df['Embarked'] = df['Embarked'].map(dict_embarked)
except:
    pass

# 4. YENİ BİR DEĞİŞKEN OLUŞTURALIM: "AİLE BÜYÜKLÜĞÜ"
# Kardeş + Ebeveyn + Kendisi = Aile Büyüklüğü
df['Aile_Buyuklugu'] = df['SibSp'] + df['Parch'] + 1

# Görsel Teması
sns.set_theme(style="whitegrid")

# --- İLERİ SEVİYE ANALİZLER ---

# ANALİZ 1: 3 Boyutlu Analiz (Sınıf + Cinsiyet + Hayatta Kalma)
# "Fakir kadınlar mı daha şanslıydı, zengin erkekler mi?" sorusunun cevabı.
g = sns.catplot(
    data=df, kind="bar",
    x="Pclass", y="Survived", hue="Sex",
    palette="dark", alpha=.6, height=6
)
# (Not: Y ekseni hayatta kalma oranını gösterir. Hata çubukları güven aralığıdır)
plt.title('Sınıf ve Cinsiyete Göre Hayatta Kalma Oranları')
plt.ylabel('Hayatta Kalma Oranı (1.0 = %100)')
plt.show()

# ANALİZ 2: Keman Grafiği (Violin Plot) - Yaş, Sınıf ve Cinsiyet
# Yaş dağılımını "keman" şeklinde gösterir. Şişman yerler yoğunluğu anlatır.
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x="Pclass", y="Age", hue="Survived",
               split=True, inner="quart", palette="muted")
plt.title('Sınıflara Göre Yaş Dağılımı ve Hayatta Kalma (Keman Grafiği)')
plt.show()
# ANALİZ 3: Aile Büyüklüğü Analizi
# "Yalnızlar mı öldü, kalabalık aileler mi?"
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='Aile_Buyuklugu', y='Survived', color='r')  # Hata düzeltildi: hue kaldırıldı
plt.title('Aile Büyüklüğü Arttıkça Hayatta Kalma Şansı Nasıl Değişiyor?')
plt.ylabel('Hayatta Kalma Oranı')
plt.xlabel('Ailedeki Kişi Sayısı')
plt.show()

# ANALİZ 4: Pairplot (BÜTÜN İLİŞKİLER)
# Sayısal olan her şeyi birbiriyle kıyaslar. (Yaş vs Fiyat, Aile vs Yaş vb.)
# Bu grafik biraz geç açılabilir, bekle.
sns.pairplot(df[['Age', 'Fare', 'SibSp', 'Parch', 'Aile_Buyuklugu', 'Survived']],
             hue='Survived', palette='husl')
plt.show()

# ANALİZ 5: Gelişmiş Korelasyon Matrisi (Her şeyin özeti)
plt.figure(figsize=(12, 10))
# Sadece sayısal verileri alıyoruz
numeric_df = pd.read_csv('titanic.csv').select_dtypes(include=['number'])
# Korelasyonu hesapla
corr = numeric_df.corr()
# Isı haritasını çiz
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Bütün Değişkenlerin Birbiriyle İlişkisi (Korelasyon)')
plt.show()