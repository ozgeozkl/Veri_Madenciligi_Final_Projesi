# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:31:07 2026

@author: öz
"""

import pandas as pd
import warnings
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Konsolu temiz tutmak için gereksiz uyarıları kapatıyoruz
warnings.filterwarnings('ignore')

# 1. Veriyi Yükleme ve Kodlama (One-Hot Encoding)
print("1. Veri yükleniyor ve metinler sayılara çevriliyor...")
df = pd.read_csv("Temizlenmis_Ogrenci_Verisi.csv")
kategorik_sutunlar = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=kategorik_sutunlar, drop_first=True)

# 2. X (Girdiler) ve y (Hedef) Olarak Ayırma
y = df_encoded['Target_Success']
X = df_encoded.drop('Target_Success', axis=1)

# 3. Veri Ölçeklendirme (Standardizasyon)
# MLP ve Lojistik Regresyon gibi modellerin matematiğinin bozulmaması için
# tüm sayıları (çalışma saati, uyku saati vb.) aynı ölçeğe getiriyoruz.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Yönergede İstenen 3 Modeli Tanımlama
modeller = {
    "Lojistik Regresyon": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest (Rastgele Orman)": RandomForestClassifier(random_state=42, n_estimators=100),
    "MLP (Çok Katmanlı Sinir Ağı)": MLPClassifier(random_state=42, hidden_layer_sizes=(50,), max_iter=500)
}

# 5. Modelleri 10-Fold Cross Validation ile Test Etme
istenen_metrikler = ['accuracy', 'f1', 'roc_auc']

print("\n2. Modeller 10-Fold Cross-Validation ile yarışa başlıyor!")
print("(Bu işlem bilgisayarın hızına göre 10-30 saniye sürebilir, lütfen bekleyin...)\n")
print("-" * 50)

for isim, model in modeller.items():
    # Modeli 10 kez sınava sokuyoruz
    sonuclar = cross_validate(model, X_scaled, y, cv=10, scoring=istenen_metrikler)
    
    # 10 sınavın ortalama notlarını hesaplıyoruz
    ortalama_accuracy = sonuclar['test_accuracy'].mean()
    ortalama_f1 = sonuclar['test_f1'].mean()
    ortalama_roc_auc = sonuclar['test_roc_auc'].mean()
    
    # Sonuçları konsola yazdırıyoruz
    print(f"🥇 Model: {isim}")
    print(f"   Doğruluk (Accuracy) : %{ortalama_accuracy * 100:.2f}")
    print(f"   F1-Skoru            : %{ortalama_f1 * 100:.2f}")
    print(f"   ROC-AUC             : %{ortalama_roc_auc * 100:.2f}")
    print("-" * 50)
    
print("\nİşlem Başarıyla Tamamlandı! Bu sonuçları raporunuza ekleyebilirsiniz.")