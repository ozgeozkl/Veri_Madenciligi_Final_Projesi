# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:35:39 2026

@author: öz
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Temizlenmiş veriyi yüklüyoruz
df = pd.read_csv("Temizlenmis_Ogrenci_Verisi.csv")

# 2. Pasta Grafiği (Sınıf Dengesi)
plt.figure(figsize=(6, 6))
df['Target_Success'].value_counts().plot.pie(
    autopct='%1.1f%%', 
    colors=['#4CAF50', '#F44336'], 
    labels=['Başarılı (1)', 'Riskli (0)'],
    startangle=90
)
plt.title("Öğrenci Başarı Durumu Dağılımı")
plt.ylabel("")
plt.savefig("eda_pasta.png") # Resmi masaüstüne kaydeder
plt.show()

# 3. Korelasyon Matrisi (Isı Haritası)
plt.figure(figsize=(10, 8))
sayisal_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(sayisal_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Öznitelikler Arası Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("eda_korelasyon.png") # Resmi masaüstüne kaydeder
plt.show()

print("Grafikler 'eda_pasta.png' ve 'eda_korelasyon.png' adıyla başarıyla kaydedildi!")