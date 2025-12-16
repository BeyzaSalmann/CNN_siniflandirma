# CNN_siniflandirma
# UNO Kartları Sınıflandırma Projesi

Bu proje, **Bilgisayar Görmesi (Computer Vision)** ve **Derin Öğrenme (Deep Learning)** teknikleri kullanılarak, UNO oyun kartları üzerindeki sembollerin (**Engel/Skip** ve **Yön Değiştir/Reverse**) sınıflandırılması amacıyla geliştirilmiştir.

Proje kapsamında özgün bir veri seti oluşturulmuş ve üç farklı model mimarisi (Transfer Learning, Temel CNN, Geliştirilmiş CNN) tasarlanarak performansları karşılaştırılmıştır. Özellikle Model 3 aşamasında, temel modelin eksikleri analiz edilerek **5 aşamalı bir optimizasyon süreci** izlenmiştir.

---

## 📂 Veri Seti (Dataset)
* **Toplama Yöntemi:** Veriler farklı açılardan mobil cihaz kamerası ile özgün olarak toplanmıştır.
* **Sınıflar:**
    * `engel` (Skip )
    * `yon_degistir` (Reverse )
* **Ön İşleme:**
    * Tüm görüntüler **128x128 piksel** boyutuna getirilmiştir.
    * Piksel değerleri 0-255 aralığından **0-1 aralığına normalize** edilmiştir.
    * Model 3 aşamasında **Veri Artırma (Data Augmentation)** teknikleri uygulanmıştır.

## 🛠️ Kurulum ve Gereksinimler

Bu proje **Google Colab** bulut tabanlı geliştirme ortamında hazırlanmıştır. Projeyi sorunsuz çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### Gerekli Kütüphaneler
Projenin çalışması için aşağıdaki Python kütüphanelerine ihtiyaç vardır (Google Colab'da bunlar varsayılan olarak yüklüdür):
* `tensorflow` (Derin Öğrenme altyapısı)
* `keras` (Model mimarisi için)
* `numpy` (Matematiksel işlemler)
* `pandas` (Veri analizi ve tablolar)
* `matplotlib` (Grafik çizimi)
* `opencv-python` (Görüntü işleme)

### Çalıştırma Adımları

** Google Colab**
1. Bu repodaki `.ipynb` uzantılı dosyaları ve `dataset` klasörünü (veya `veri_seti.zip` dosyasını) Google Drive'ınıza yükleyin.
2. Dosyaları Google Colab ile açın.
3. Dosyanın başındaki Google Drive bağlantı kodunu çalıştırın.
4. Sırasıyla tüm hücreleri çalıştırarak eğitimi başlatın.

---

## Geliştirilen Modeller ve Yöntemler

### 1. Model 1: Transfer Learning (VGG16) ile Referans Başarımı
Projenin ilk aşamasında, elimizdeki veri setinin küçük olması (sınıf başına yaklaşık 50 görsel) nedeniyle, derin mimarileri sıfırdan eğitmek yerine literatürde başarısı kanıtlanmış **Transfer Öğrenme (Transfer Learning)** stratejisi benimsenmiştir. ImageNet ağırlıklarıyla eğitilmiş **VGG16** mimarisi kullanılarak, sınırlı veriye rağmen **%70** referans başarımı elde edilmiştir.

### 2. Model 2: Temel CNN (Baseline) Tasarımı
İkinci aşamada, hazır bir modelin gücünü kullanmadan, tamamen sıfırdan (from scratch) eğitilen özgün bir yapı kurulmuştur. 3 ardışık evrişim bloğundan (32-64-128 filtre) oluşan bu temel modelde herhangi bir veri artırma uygulanmamıştır. Sonuç olarak modelin eğitim verisini ezberlediği (Overfitting) gözlemlenmiş ve test başarımı **%55** seviyesinde kalmıştır.

### 3. Model 3: Geliştirilmiş CNN ve Optimizasyon Süreci (Final Model)
Projenin en kapsamlı aşamasıdır. Model 2'nin düşük performansını gidermek için **5 aşamalı deneysel bir süreç** izlenmiştir:
1.  **Veri Artırma (Data Augmentation):** Görüntüler döndürülerek ve kaydırılarak veri çeşitliliği artırıldı.
2.  **Dropout Eklemesi:** Ağın kararlılığını artırmak için %30 ve %50 oranlarında Dropout katmanları eklendi.
3.  **Mimari Sadeleştirme:** Filtre sayıları (16-32-64) optimize edilerek işlem yükü hafifletildi.
4.  **Hiperparametre Ayarı:** Öğrenme oranı (Learning Rate) `0.001`'den `0.0001`'e düşürülerek hassas eğitim (fine-tuning) sağlandı.

Bu adımlar sonucunda, Model 2'nin performansı ciddi oranda aşılarak **%80** başarıya ulaşılmış ve model kararlı hale getirilmiştir. *(Detaylı deney tablosu `model3.ipynb` dosyasında mevcuttur.)*

---

##  Performans Karşılaştırması

Üç modelin test seti üzerindeki nihai başarımları aşağıdaki gibidir:

| Model | Yöntem | Veri Artırma | Test Başarısı (Accuracy) |
|-------|--------|--------------|--------------------------|
| **Model 1** | Transfer Learning (VGG16) | Hayır | **%70** |
| **Model 2** | Temel CNN (Custom) | Hayır | **%55** |
| **Model 3** | **Geliştirilmiş CNN (Optimize)** | **Evet** | **%80**  |
