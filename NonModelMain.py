import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage # Median filter ve morfoloji için
import os
from PIL import Image


def advanced_preprocess(image_path):
    # 1. Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gürültü Temizleme (Hibrit Yöntem)
    # Median Filter: Ani parlamaları ve küçük objeleri (ev, araba) yok eder
    median_filtered = cv2.medianBlur(gray, 5)
    
    # Gaussian Blur: Genel yüzeyi yumuşatır, konturların (eğrilerin) akıcı olmasını sağlar
    final_gray = cv2.GaussianBlur(median_filtered, (7, 7), 0)
    
    return img_rgb, final_gray

# --- Çalıştıralım ---
img_path = 'data/test_uydu11.png'
original, processed = advanced_preprocess(img_path)

if original is not None:
    print("Adım 2 Tamamlandı: Görüntü pürüzsüzleştirildi.")
    
def get_water_mask(img_rgb):
    # RGB'den HSV'ye geçiyoruz (Renkleri ayırmak için en iyi yöntem)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Google Earth su tonları (Koyu mavi/Lacivert) için maske aralığı
    # Bu değerleri görüntüne göre (110, 50, 50) gibi oynatabilirsin
    lower_blue = np.array([90, 40, 20]) 
    upper_blue = np.array([135, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Maskedeki delikleri kapat (Küçük adacıkları temizle)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# Maskeyi alalım
water_mask = get_water_mask(original)

# Görselleştirme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("İşlenmiş Gri Görüntü")
plt.imshow(processed, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Tespit Edilen Su (Maske)")
plt.imshow(water_mask, cmap='Blues')
plt.show()


def generate_topography(processed_gray, water_mask):
    # 1. Ham Yükseklik Modeli (DEM) Tahmini
    # Görüntüyü 0-255 arasından 0.0 - 1.0 arasına normalize edelim
    dem = processed_gray.astype(float) / 255.0
    
    # 2. Su Düzleme (Water Flattening)
    # Su maskesinin olduğu yerleri (255 olan pikseller) 0 yüksekliğine çekiyoruz
    dem[water_mask > 0] = 0.0
    
    # 3. Kontur (Eş Yükselti Eğrisi) Üretimi
    # Matplotlib'in contour fonksiyonu bizim için izohipsleri çizecek
    plt.figure(figsize=(12, 8))
    
    # Arka plana renkli yükseklik haritasını koyalım (Terrain: Coğrafi renkler)
    plt.imshow(dem, cmap='terrain', origin='upper')
    plt.colorbar(label='Tahmini Rölatif Yükseklik')
    
    # Üzerine kontur çizgilerini ekleyelim
    # levels=20 -> haritada 20 tane eş yükselti çizgisi olsun demek
    contours = plt.contour(dem, levels=15, colors='black', linewidths=0.5, alpha=0.7)
    
    # Çizgilerin üzerine yükseklik değerlerini yazdıralım
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    plt.title("Üretilen Topolojik Harita (Eş Yükselti Eğrileri)")
    plt.axis('off')
    plt.show()
    
    return dem

# --- ANA ÇALIŞTIRMAYA EKLE ---
if original is not None:
    dem_result = generate_topography(processed, water_mask)
    print("Adım 4 ve 5 Tamamlandı: Topolojik harita üretildi!")
    

def save_outputs(original_img, dem_data, output_folder='output'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 1. Tahmini DEM verisini (0-255 arası) PNG olarak kaydet
    dem_to_save = (dem_data * 255).astype(np.uint8)
    cv2.imwrite(f'{output_folder}/predicted_dem.png', dem_to_save)
    
    # 2. Final Topolojik Haritayı (Figürü) Kaydet
    # Not: plt.savefig o an açık olan en son figürü kaydeder.
    plt.figure(figsize=(12, 8))
    plt.imshow(dem_data, cmap='terrain')
    contours = plt.contour(dem_data, levels=15, colors='black', linewidths=0.5)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.axis('off')
    plt.savefig(f'{output_folder}/final_topolojik_harita.png', dpi=300, bbox_inches='tight')
    
    print(f"Çıktılar '{output_folder}' klasörüne kaydedildi.")

# --- ANA ÇALIŞTIRMAYA EKLE ---
save_outputs(original, dem_result)