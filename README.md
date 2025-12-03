# CNN_siniflandirma
# UNO Kartları Sınıflandırma Projesi 

Bu proje, **Bilgisayar Görmesi (Computer Vision)** ve **Derin Öğrenme (Deep Learning)** teknikleri kullanılarak, UNO oyun kartları üzerindeki sembollerin (**Engel/Skip** ve **Yön Değiştir/Reverse**) sınıflandırılması amacıyla geliştirilmiştir.

Proje kapsamında özgün bir veri seti oluşturulmuş ve üç farklı model mimarisi (Transfer Learning, Temel CNN, Geliştirilmiş CNN) tasarlanarak performansları karşılaştırılmıştır.

---

##  Veri Seti (Dataset)
* **Toplama Yöntemi:** Veriler farklı açılardan mobil cihaz kamerası ile özgün olarak toplanmıştır.
* **Sınıflar:**
    * `engel` (Skip )
    * `yon_degistir` (Reverse )
* **Ön İşleme:**
    * Tüm görüntüler **128x128 piksel** boyutuna getirilmiştir.
    * Piksel değerleri 0-255 aralığından **0-1 aralığına normalize** edilmiştir.
    * Model 3 aşamasında **Veri Artırma (Data Augmentation)** teknikleri uygulanmıştır.

---

## Geliştirilen Modeller ve Yöntemler

### 1. Model 1: Transfer Learning (VGG16) ile Referans Başarımı
Projenin ilk aşamasında, elimizdeki veri setinin küçük olması (sınıf başına yaklaşık 50 görsel) nedeniyle, derin mimarileri sıfırdan eğitmek yerine literatürde başarısı kanıtlanmış **Transfer Öğrenme (Transfer Learning)** stratejisi benimsenmiştir. Bu bağlamda, ImageNet veri seti üzerinde milyonlarca görselle eğitilmiş olan **VGG16** mimarisi temel alınmıştır. Modelin öznitelik çıkarma (feature extraction) katmanları dondurularak, önceden öğrenilmiş kenar ve doku bilgileri korunmuş; çıkışına ise projeye özgü sınıflandırma katmanları eklenmiştir. Bu yaklaşım sayesinde, sınırlı veriye rağmen modelin ezberlemesi (overfitting) engellenmiş ve **%90** gibi yüksek bir doğruluk oranına ulaşılarak projenin referans başarımı belirlenmiştir.

### 2. Model 2: Temel CNN (Baseline) Tasarımı
İkinci aşamada, hazır bir modelin gücünü kullanmadan, tamamen sıfırdan (from scratch) eğitilen özgün bir yapay sinir ağı mimarisinin performansı test edilmiştir. CIFAR-10 gibi klasik problemlerde kullanılan yapı referans alınarak; 3 ardışık evrişim bloğundan (Conv2D + MaxPooling) oluşan, filtre sayılarının 32'den 128'e kademeli olarak arttığı temel bir **CNN mimarisi** tasarlanmıştır. Bu aşamada modelin "saf" performansını gözlemlemek amacıyla herhangi bir veri artırma işlemi uygulanmamış, sadece veri normalizasyonu yapılmıştır. Beklendiği üzere, veri azlığı nedeniyle bu model **%55** bir başarı göstererek geliştirilmeye açık bir zemin oluşturmuştur.

### 3. Model 3: Geliştirilmiş ve Optimize Edilmiş CNN
Son aşamada, Model 2'nin performansını artırmak ve daha kararlı bir yapı elde etmek için kapsamlı optimizasyonlar yapılmıştır. İlk olarak, veri setindeki kısıtlı sayıyı telafi etmek için **Veri Artırma (Data Augmentation)** teknikleri devreye alınmış; eğitim verileri sanal olarak döndürülerek, yakınlaştırılarak ve aynalanarak çoğaltılmıştır. Mimari tarafta ise "Az Veri, Sade Model" prensibiyle filtre sayıları optimize edilmiş (16-32-64 yapısına geçilmiş), Dropout oranı düşürülerek modelin hafıza kapasitesi artırılmış ve öğrenme hızı (Learning Rate) daha hassas bir eğitim için revize edilmiştir. Bu stratejik hamleler sonucunda Model 2'nin performansı belirgin şekilde aşılarak **%80** seviyesine çıkarılmış ve projenin "iyileştirme" hedefi başarıyla gerçekleştirilmiştir.

-------------------------------------------------------------------------------------------------------------------------------------------------------

##  Performans Karşılaştırması

Üç modelin test seti üzerindeki nihai başarımları aşağıdaki gibidir:

| Model | Yöntem | Veri Artırma | Test Başarısı (Accuracy) | Sonuç Yorumu |
|-------|--------|--------------|--------------------------|--------------|
| **Model 1** | Transfer Learning (VGG16) | Hayır | **%90** | En yüksek başarı (Hazır ağırlık avantajı). |
| **Model 2** | Temel CNN (Custom) | Hayır | **%55** | Temel seviye başarı, geliştirilmeye açık. |
| **Model 3** | Geliştirilmiş CNN | **Evet** | **%80** | Veri artırma ve mimari optimizasyon ile Model 2'ye göre artış sağlanmıştır. |





