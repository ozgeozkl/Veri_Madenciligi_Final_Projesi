import pandas as pd
import warnings
from sklearn.model_selection import cross_validate, GridSearchCV
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. GridSearchCV ile Random Forest Optimizasyonu
print("\n🔍 Random Forest için GridSearchCV ile en iyi parametreler aranıyor...")
print("(Bu işlem bilgisayarın hızına göre birkaç saniye sürebilir, lütfen bekleyin...)")

# Modeli en iyi ayarlarla bulmak için bir arama uzayı (param_grid) tanımlıyoruz
rf_parametreleri = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

# 5-Fold Cross Validation ile en iyi parametreleri arıyoruz
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_parametreleri, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y)
en_iyi_rf = grid_search.best_estimator_

print(f"✅ En İyi Parametreler Bulundu: {grid_search.best_params_}\n")

# 5. Yönergede İstenen Modelleri Tanımlama (Artık Optimize Edilmiş RF kullanıyoruz)
modeller = {
    "Lojistik Regresyon": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest (GridSearch Optimize Edilmiş)": en_iyi_rf,
    "MLP (Çok Katmanlı Sinir Ağı)": MLPClassifier(random_state=42, hidden_layer_sizes=(50,), max_iter=500)
}

# 6. Modelleri 10-Fold Cross Validation ile Test Etme
istenen_metrikler = ['accuracy', 'f1', 'roc_auc']

print("2. Modeller 10-Fold Cross-Validation ile yarışa başlıyor!")
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
    
print("\nİşlem Başarıyla Tamamlandı! Bu ekranın görüntüsünü 'gridsearch.png' adıyla kaydedip Overleaf'e yükleyebilirsiniz.")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("\n📊 Şampiyon Model (Lojistik Regresyon) için Confusion Matrix çizdiriliyor...")
# Lojistik Regresyonun 10-Fold CV üzerindeki tahminlerini alıyoruz
y_tahmin = cross_val_predict(modeller["Lojistik Regresyon"], X_scaled, y, cv=10)

# Matrisi oluşturup çizdiriyoruz
cm = confusion_matrix(y, y_tahmin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Riskli (0)', 'Başarılı (1)'])

plt.figure(figsize=(6,6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Lojistik Regresyon - Karmaşıklık Matrisi (Hata Analizi)")
plt.savefig("confusion_matrix.png") # Resmi masaüstüne kaydeder
plt.show()

print("✅ 'confusion_matrix.png' başarıyla kaydedildi! ")
