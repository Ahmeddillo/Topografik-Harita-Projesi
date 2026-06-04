import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. ADIM: MiDaS Yapay Zeka Modelini Hazırla
# ==========================================
def load_ai_model():
    print("AI Modeli yükleniyor (MiDaS)...")
    model_type = "MiDaS_small" # Hız için small seçtik
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    # Modelin beklediği görüntü dönüştürücüleri
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
    
    return midas, transform, device

# ==========================================
# 2. ADIM: Su Maskesi (Senin Klasik Metodun)
# ==========================================
def get_water_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Su için mavi tonları aralığı
    lower_blue = np.array([90, 40, 20])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask

# ==========================================
# 3. ADIM: Ana İşleme Akışı
# ==========================================
def process_satellite_image(img_path):
    # Klasör Kontrolü
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' klasörü oluşturuldu.")

    # 1. Görüntüyü oku
    img = cv2.imread(img_path)
    if img is None:
        print("Hata: Dosya bulunamadı!")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. AI Modelini Çağır ve Tahmin Yap
    model, transform, device = load_ai_model()
    input_batch = transform(img_rgb).to(device)
    
    print("AI yükseklik tahmini yapıyor...")
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    dem_ai = prediction.cpu().numpy()
    dem_ai = (dem_ai - dem_ai.min()) / (dem_ai.max() - dem_ai.min())
    
    # 3. Su Maskesi Uygula
    water_mask = get_water_mask(img_rgb)
    dem_ai[water_mask > 0] = 0 
    
    # 4. Görselleştirme ve KAYDETME
    fig = plt.figure(figsize=(15, 10))
    
    # Orijinal Uydu
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Orijinal Uydu Görüntüsü")
    plt.axis('off')
    
    # AI Destekli Topoğrafik Harita
    plt.subplot(1, 2, 2)
    plt.imshow(dem_ai, cmap='terrain')
    contours = plt.contour(dem_ai, levels=20, colors='black', linewidths=0.5, alpha=0.6)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    plt.colorbar(label='Göreli Yükseklik (AI Tahmini)')
    plt.title("AI + Klasik Maskeleme Haritası")
    plt.axis('off')
    
    plt.tight_layout()
    
    # --- DOSYAYI KAYDET ---
    file_name = os.path.basename(img_path).split('.')[0] # Dosya adını al (test_uydu gibi)
    save_path = os.path.join(output_folder, f"{file_name}_topolojik_harita.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Ayrıca sadece yükseklik verisini (DEM) grayscale olarak da kaydedelim (Opsiyonel)
    dem_gray_save = (dem_ai * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"{file_name}_DEM.png"), dem_gray_save)
    
    print(f"Sonuçlar başarıyla kaydedildi: {save_path}")
    plt.show()

# PROGRAMI ÇALIŞTIR
# Kendi dosya yolunu buraya yaz:
process_satellite_image('data/test_uydu10.png')