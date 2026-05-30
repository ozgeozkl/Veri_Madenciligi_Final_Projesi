# Öğrenci Performans Faktörleri Sınıflandırma Analizi

Bu proje, öğrencilerin çalışma alışkanlıkları, sosyo-ekonomik durumları ve aile faktörleri gibi verileri kullanarak akademik başarılarını (Riskli/Başarılı) önceden tahmin etmeyi amaçlayan bir Veri Madenciliği (Data Mining) çalışmasıdır. Proje süreçleri endüstri standardı olan CRISP-DM metodolojisine uygun olarak yürütülmüştür.

## Kullanılan Teknolojiler ve Algoritmalar
* **Dil & Kütüphaneler:** Python (Pandas, Scikit-Learn, Matplotlib, Seaborn)
* **Veri Ön İşleme:** Imputation (Mod), One-Hot Encoding, StandardScaler
* **Makine Öğrenmesi Modelleri:** Lojistik Regresyon, Random Forest, Çok Katmanlı Sinir Ağı (MLP)
* **Optimizasyon:** GridSearchCV (Hiperparametre optimizasyonu)
* **Değerlendirme:** 10-Fold Cross-Validation, Confusion Matrix, Accuracy, F1-Score, ROC-AUC

## Proje Çıktıları
Yapılan 10-Fold Cross-Validation testleri sonucunda en yüksek performansı **Lojistik Regresyon** (%98.03 Doğruluk, %99.51 ROC-AUC) göstermiştir. Random Forest modelindeki olası ezberleme (Overfitting - Yüksek Varyans) sorunları Bias-Variance ikilemi üzerinden incelenmiş ve raporlanmıştır.

## Repo İçeriği
* `modelleme.py` ve `eda_grafikleri.py`: Veri işleme, model eğitimi ve görselleştirme kodları.
* `Temizlenmis_Ogrenci_Verisi.csv`: Modele hazır hale getirilmiş, encode ve scale edilmiş veri seti.
* `*.png` dosyaları: Keşifçi Veri Analizi (EDA) ve Confusion Matrix çıktıları.
* `veri_madenciligi_final_projesi.pdf`: LaTeX ile yazılmış kapsamlı 13 sayfalık akademik final raporu.
