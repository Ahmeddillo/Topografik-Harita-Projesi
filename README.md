# 🛰️ Uydu Görüntüsünden Otomatik Topografik Harita Üretimi

<img width="841" height="503" alt="Ekran görüntüsü 2026-06-04 223929" src="https://github.com/user-attachments/assets/f319f1a1-3025-4518-92cd-aa1e53026139" />


Uydu fotoğraflarını yapay zeka ile analiz ederek 2D kontur haritası ve 3D arazi yüzeyi üreten bir Streamlit uygulaması.

---

## 🚀 Özellikler

- **MiDaS AI Derinlik Tahmini** — Intel'in MiDaS Small modeliyle görüntüden göreli yükseklik haritası (DEM) üretimi
- **Su Bölgesi Tespiti** — HSV renk uzayı ve konveks gövde filtresiyle otomatik su alanı maskeleme
- **Arazi Sınıflandırması** — Bitki örtüsü, kıyı ve kentsel alan tespiti
- **Gölge Telafisi** — CLAHE kontrast iyileştirme ve eğim bazlı gölge düzeltmesi
- **2D Topografik Harita** — Hypsometrik renk paleti, hillshade gölgelendirmesi ve kontur çizgileri
- **3D Yüzey Görünümü** — İsteğe bağlı interaktif 3D arazi modeli
- **Çoklu Dışa Aktarım** — PNG, gri ton DEM ve GeoTIFF formatlarında indirme desteği

---

## 🖥️ Kurulum

```bash
pip install streamlit opencv-python numpy matplotlib scipy torch torchvision Pillow yt-dlp
```

GeoTIFF dışa aktarımı için ek olarak:

```bash
pip install rasterio
```

---

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

---

## 📖 Kullanım

1. Uygulamayı tarayıcıda açın
2. PNG, JPG veya TIFF formatında bir uydu görüntüsü yükleyin
3. Kenar çubuğundan kontur aralığı ve görünüm seçeneklerini ayarlayın
4. **ANALİZİ BAŞLAT** butonuna tıklayın
5. Sonuçları istediğiniz formatta indirin

---

## ⚙️ Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| Kontur Aralığı | 0.04 | Küçük değer = daha sık kontur çizgisi |
| 3D Yüzey Görünümü | Açık | 3D yüzey modelini etkinleştirir |
| GeoTIFF Dışa Aktar | Kapalı | rasterio kurulu olmalıdır |
| Gölge Debug | Kapalı | Gölge maskesi ve DEM'i ayrı gösterir |

---

## 🧠 Teknik Detaylar

Yükseklik haritası üretimi üç bileşenin ağırlıklı birleşiminden oluşur:

- **MiDaS derinlik tahmini** (%65 ağırlık, gölgeli alanlarda artırılır)
- **CLAHE uygulanmış gri ton parlaklığı** (%17)
- **Sobel gradyan büyüklüğü** (%18)

Su bölgeleri tespit edildiğinde DEM'de sıfır yüksekliğe sabitlenir ve çevre piksellerine mesafe bazlı yükseklik geçişi uygulanır.

---

## 📦 Bağımlılıklar

- `streamlit`: Arayüz
- `opencv-python`: Görüntü işleme
- `torch` / `torchvision`: MiDaS modeli
- `matplotlib`: Görselleştirme
- `scipy`: Filtreler ve mesafe dönüşümü
- `Pillow`: Görüntü dışa aktarımı
- `rasterio`: *(opsiyonel)* — GeoTIFF desteği

---

## 📄 Lisans

Bu proje İşletmede Mesleki Eğitim amacıyla Ahmed Dillo tarafından gelişitirlmiştir.
