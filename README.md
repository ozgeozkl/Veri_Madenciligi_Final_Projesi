# Öğrenci Performans Faktörleri - Veri Madenciliği Projesi

Bu proje, BLM308 Veri Madenciliği dersi final ödevi kapsamında geliştirilmiş uçtan uca bir makine öğrenmesi uygulamasıdır. Projede CRISP-DM metodolojisi takip edilmiştir.

## Projenin Amacı
Okul yönetiminin, öğrencilerin performans faktörlerini (çalışma saati, uyku, aile ilgisi vb.) analiz ederek, risk altındaki öğrencileri sınavlar yapılmadan önce tespit etmesini sağlayan 3 farklı yapay zeka modelinin karşılaştırılmasıdır. 

## Klasör Yapısı
* `StudentPerformanceFactors (1).csv`: Kaggle üzerinden elde edilen ham veri seti.
* `Temizlenmis_Ogrenci_Verisi.csv`: Mod ile eksik verileri doldurulmuş ve One-Hot Encoding uygulanmış, makine öğrenmesine hazır veri seti.
* `veri_hazirligi.py`: Veri ön işleme ve temizleme adımlarını içeren Python betiği.
* `modelleme.py`: Modellerin kurulduğu, Cross-Validation (Çapraz Doğrulama) ve performans metriklerinin hesaplandığı ana dosya.

## Kullanılan Modeller ve Sonuçlar
Veri seti 10-Fold Cross Validation ile test edilmiş ve aşağıdaki doğruluk (Accuracy) oranları elde edilmiştir:
1. **Lojistik Regresyon:** %98.03 
2. **MLP (Çok Katmanlı Sinir Ağı):** %96.76
3. **Random Forest:** %89.90

## Nasıl Çalıştırılır?
1. Repoyu bilgisayarınıza klonlayın.
2. Gerekli kütüphaneleri yükleyin: `pip install pandas scikit-learn`
3. Sırasıyla `veri_hazirligi.py` ve `modelleme.py` dosyalarını çalıştırın.
