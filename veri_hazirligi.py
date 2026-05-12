# -*- coding: utf-8 -*-
"""
Created on Wed May 13 00:11:08 2026

@author: öz
"""

import pandas as pd

# 1. Veri Setini Yükleme
print("Veri seti yükleniyor...")
df = pd.read_csv("StudentPerformanceFactors (1).csv")

# 2. Eksik Verileri Doldurma (Imputation)
print("Eksik veriler temizleniyor...")
# Boşlukları (NaN) en çok tekrar eden değer (mode) ile dolduruyoruz.
df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])

# 3. Hedef Sınıfı (Target Class) Oluşturma
print("Hedef sınıf oluşturuluyor...")
# Öğrenci notu 67 ve üzeriyse 1 (Başarılı), altındaysa 0 (Riskli)
df['Target_Success'] = df['Exam_Score'].apply(lambda x: 1 if x >= 67 else 0)

# 4. Modeli yanıltmaması için orijinal not sütununu (Data Leakage'i önlemek için) siliyoruz.
df = df.drop('Exam_Score', axis=1)

# 5. Temizlenmiş Veriyi Kaydetme
output_file = "Temizlenmis_Ogrenci_Verisi.csv"
df.to_csv(output_file, index=False)

print(f"\nİşlem Tamam! Temizlenmiş veriniz '{output_file}' olarak kaydedildi.")
print("Yeni verinin boyutu:", df.shape)
print("\nHedef Sınıf Dağılımı:")
print(df['Target_Success'].value_counts())