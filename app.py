# app.py
# "C:\Users\EXCALIBUR\AppData\Local\spyder-6\envs\spyder-runtime\Scripts\streamlit.exe" run app.py


import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, distance_transform_edt, median_filter
import torch
import tempfile
import os
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# Sayfa ayarları
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uydu Görüntüsü → Topografik Haritası | v7",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Genel arka plan */
.stApp {
    background: #0f1117;
    color: #e8e4dc;
}

/* Başlık bandı */
.hero-band {
    background: linear-gradient(135deg, #0b1e3d 0%, #122b52 60%, #0d3a2e 100%);
    border-bottom: 2px solid #2a7a5c;
    padding: 2rem 2.5rem 1.6rem;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0 0 18px 18px;
}
.hero-band h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.5px;
    color: #e8f5f0;
    margin: 0 0 0.3rem 0;
}
.hero-band p {
    font-size: 0.92rem;
    color: #8ab9a4;
    margin: 0;
}

/* Adım kartları */
.step-card {
    background: #161b27;
    border: 1px solid #253352;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.step-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #2a7a5c;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.step-title {
    font-weight: 600;
    font-size: 1rem;
    color: #dde8e3;
}

/* Metrik kutular */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: #111827;
    border: 1px solid #1f3050;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    color: #3ecf8e;
    font-weight: 700;
}
.metric-box .lbl {
    font-size: 0.72rem;
    color: #667c93;
    margin-top: 0.15rem;
}

/* Sonuç sekme başlıkları */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    color: #667c93;
}
.stTabs [aria-selected="true"] {
    color: #3ecf8e !important;
    border-bottom: 2px solid #3ecf8e !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid #1e2d42;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #8ab9a4;
    font-size: 0.85rem;
}

/* Upload alanı */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a4060 !important;
    border-radius: 12px !important;
    background: #0e1724 !important;
    padding: 1.5rem !important;
}

/* Progress / spinner */
.stSpinner > div {
    border-top-color: #3ecf8e !important;
}

/* Buton */
.stButton > button {
    background: linear-gradient(135deg, #2a7a5c, #1a5c45);
    color: #e8f5f0;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 1.5px;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* İnfo kutusu */
.info-chip {
    display: inline-block;
    background: #122b52;
    color: #7ec8e3;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin: 0.2rem 0.2rem 0 0;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-band">
  <h1>🛰️ Uydu Görüntüsü → Topografik Haritası</h1>
  <p>MiDaS AI derinlik modeli · Hypsometrik renk paleti · 2D kontur + 3D yüzey &nbsp;|&nbsp; <b>v7</b></p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar — Parametreler
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametreler")
    st.markdown("---")

    contour_interval = st.slider(
        "Kontur Aralığı",
        min_value=0.02, max_value=0.15, value=0.04, step=0.01,
        help="Küçük değer = daha sık kontur çizgisi"
    )
    save_3d = st.toggle("3D Yüzey Görünümü", value=True)
    save_geotiff_flag = st.toggle("GeoTIFF Dışa Aktar", value=False,
                                   help="rasterio kurulu olmalıdır")
    debug_shadow = st.toggle("Gölge Debug Görünümü", value=False)

    st.markdown("---")
    st.markdown("**Model:** MiDaS Small")
    st.markdown("**Cihaz:** CUDA varsa GPU, yoksa CPU")
    st.markdown("---")
    st.markdown("""
<span class="info-chip">HSV su tespiti</span>
<span class="info-chip">Konveks gövde filtresi</span>
<span class="info-chip">CLAHE</span>
<span class="info-chip">Bilateral filtre</span>
<span class="info-chip">Hillshade</span>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# ORİJİNAL FONKSİYONLAR — HİÇBİR ŞEY DEĞİŞTİRİLMEDİ
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_ai_model():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return midas, midas_transforms.small_transform, device


def get_water_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([95,  60,  30]), np.array([130, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([85,  50,  80]), np.array([100, 180, 220]))
    water = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((7, 7), np.uint8)
    water  = cv2.morphologyEx(water, cv2.MORPH_OPEN,  kernel)
    water  = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    cleaned = np.zeros_like(water)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 400:
            continue
        mask_i = (labels == i).astype(np.uint8) * 255
        contours_i, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_i:
            hull_area = cv2.contourArea(cv2.convexHull(contours_i[0]))
            if hull_area > 0:
                convexity = area / hull_area
                if area < 3000 and convexity > 0.75:
                    continue
        cleaned[labels == i] = 255
    return cleaned


def classify_terrain(img_rgb, water_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    veg_mask   = cv2.inRange(hsv, np.array([35, 25, 30]),  np.array([85, 255, 255]))
    coast_mask = cv2.inRange(hsv, np.array([15,  8, 140]), np.array([35,  70, 255]))
    urban_mask = cv2.inRange(hsv, np.array([0,   0,  60]), np.array([179, 35, 210]))
    for m in [veg_mask, coast_mask, urban_mask]:
        m[water_mask > 0] = 0
    return {"water": water_mask, "vegetation": veg_mask,
            "coast": coast_mask, "urban": urban_mask}


def build_shadow_compensation(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L   = lab[:, :, 0].astype(np.float32) / 255.0
    shadow_hard = (L < 0.38).astype(np.float32)
    shadow_mask = gaussian_filter(shadow_hard, sigma=5.0)
    shadow_mask = np.clip(shadow_mask, 0, 1)
    gray_uint8  = (L * 255).astype(np.uint8)
    clahe       = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
    gray_clahe  = clahe.apply(gray_uint8).astype(np.float32) / 255.0
    return shadow_mask, gray_clahe


def estimate_slope_correction(gray, shadow_mask):
    sobel_y      = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    north_facing = np.clip(-sobel_y, 0, None)
    north_facing = (north_facing - north_facing.min()) / (north_facing.max() - north_facing.min() + 1e-8)
    return north_facing * shadow_mask * 0.10


def build_dem(img_rgb, model, transform, device, terrain_classes):
    water_mask = terrain_classes["water"]
    veg_mask   = terrain_classes["vegetation"]
    has_water  = (water_mask > 0).sum() / water_mask.size > 0.02

    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map    = prediction.cpu().numpy()
    height_midas = depth_map.max() - depth_map
    height_midas = (height_midas - height_midas.min()) / (height_midas.max() - height_midas.min())
    height_midas = median_filter(height_midas, size=5)

    shadow_mask, gray_clahe = build_shadow_compensation(img_rgb)
    slope_correction = estimate_slope_correction(gray_clahe, shadow_mask)

    sobel_x  = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=5)
    sobel_y  = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

    w_midas = 0.65 + shadow_mask * 0.15
    w_gray  = 0.17 - shadow_mask * 0.15
    w_grad  = 0.18
    w_total = w_midas + w_gray + w_grad
    w_midas /= w_total; w_gray /= w_total; w_grad /= w_total

    combined = w_midas * height_midas + w_gray * gray_clahe + w_grad * grad_mag
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    combined = np.clip(combined + slope_correction, 0, 1)

    p2, p98  = np.percentile(combined, 2), np.percentile(combined, 98)
    combined = np.clip((combined - p2) / (p98 - p2 + 1e-8), 0, 1)
    combined = gaussian_filter(combined, sigma=2.5)

    if has_water:
        water_binary    = (water_mask > 0).astype(np.float32)
        dist_from_water = distance_transform_edt(1 - water_binary)
        dist_norm       = np.clip(dist_from_water / 120.0, 0, 1) ** 0.8
        combined        = combined * dist_norm
        combined[water_mask > 0] = 0.0

    veg_boost = (veg_mask > 0).astype(np.float32) * 0.02
    combined  = np.clip(combined + veg_boost, 0, 1)
    if combined.max() > 0:
        combined = combined / combined.max()

    combined_u8 = (combined * 255).astype(np.uint8)
    combined_u8 = cv2.bilateralFilter(combined_u8, d=15, sigmaColor=75, sigmaSpace=75)
    combined    = combined_u8.astype(np.float32) / 255.0

    return combined, has_water


def get_hypsometric_cmap(has_water=True):
    if has_water:
        colors_list = [
            (0.000, "#0e2f5a"), (0.010, "#1e5c99"), (0.030, "#4a8ab5"),
            (0.055, "#c8b87a"), (0.110, "#b5a060"), (0.200, "#8fa060"),
            (0.320, "#a08855"), (0.480, "#957050"), (0.640, "#806045"),
            (0.780, "#a09080"), (0.900, "#c8bfb0"), (1.000, "#ece8e0"),
        ]
    else:
        colors_list = [
            (0.000, "#b5a060"), (0.200, "#a08855"), (0.400, "#957050"),
            (0.600, "#806045"), (0.780, "#a09080"), (0.920, "#c8bfb0"),
            (1.000, "#ece8e0"),
        ]
    return mcolors.LinearSegmentedColormap.from_list("hypsometric_v7", colors_list)


def draw_contour_map(ax, dem, has_water=True, contour_interval=0.05, title="Topolojik Harita"):
    cmap     = get_hypsometric_cmap(has_water)
    img_plot = ax.imshow(dem, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear', zorder=1)
    ls       = LightSource(azdeg=315, altdeg=45)
    hs_arr   = ls.hillshade(dem, vert_exag=4.0, dx=1, dy=1)
    ax.imshow(hs_arr, cmap='gray', vmin=0, vmax=1, interpolation='bilinear', alpha=0.22, zorder=2)
    dem_c        = gaussian_filter(median_filter(dem, size=7), sigma=2.0)
    levels       = np.arange(0, 1.0 + contour_interval, contour_interval)
    major_levels = levels[::4]
    ax.contour(dem_c, levels=levels, colors='#1a1a1a', linewidths=0.28, alpha=0.32, zorder=3)
    if len(major_levels) >= 2:
        mc = ax.contour(dem_c, levels=major_levels, colors='#0d0d0d', linewidths=1.0, alpha=0.78, zorder=4)
        ax.clabel(mc, inline=True, fontsize=6.5, fmt='%.2f', colors='#111111', inline_spacing=4)
    plt.colorbar(img_plot, ax=ax,
                 label='Göreli Yükseklik  (0 = Alçak  →  1 = Zirve)',
                 fraction=0.035, pad=0.04)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def draw_3d_surface(fig, dem, has_water=True, position=111):
    ax3d  = fig.add_subplot(position, projection='3d')
    dem_3d = gaussian_filter(dem, sigma=2.8)
    step   = max(1, dem_3d.shape[0] // 130)
    dem_s  = dem_3d[::step, ::step]
    h, w   = dem_s.shape
    X, Y   = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    cmap   = get_hypsometric_cmap(has_water)
    ls     = LightSource(azdeg=315, altdeg=45)
    rgb    = ls.shade(dem_s, cmap=cmap, vert_exag=2.0, blend_mode='soft')
    ax3d.plot_surface(X, Y, dem_s, facecolors=rgb, linewidth=0, antialiased=True,
                      rstride=1, cstride=1, alpha=0.96)
    ax3d.set_title("3D Yüzey Görünümü", fontsize=12, fontweight='bold')
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Yükseklik")
    ax3d.set_box_aspect([1, 1, 0.45])
    ax3d.view_init(elev=35, azim=-60)
    ax3d.set_facecolor('#f5f2ee')
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False


def save_geotiff(dem, output_path, bounds=(26.0, 39.0, 27.0, 40.0), crs="EPSG:4326"):
    try:
        import rasterio
        from rasterio.transform import from_bounds
        h, w    = dem.shape
        tf      = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
        dem_u16 = (dem * 65535).astype(np.uint16)
        with rasterio.open(output_path, 'w', driver='GTiff',
                           height=h, width=w, count=1,
                           dtype=dem_u16.dtype, crs=crs,
                           transform=tf, compress='lzw') as dst:
            dst.write(dem_u16, 1)
        return True
    except ImportError:
        return False

# ─────────────────────────────────────────────────────────────────
# YÜKLEME ALANI
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-num">ADIM 01</div>
  <div class="step-title">Uydu görüntüsü yükle</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "PNG, JPG veya TIFF formatında uydu fotoğrafı yükleyin",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # Görseli oku
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W       = img_rgb.shape[:2]

    # Önizleme + meta
    col_prev, col_meta = st.columns([2, 1])
    with col_prev:
        st.image(img_rgb, caption="Yüklenen Uydu Görüntüsü", use_column_width=True)
    with col_meta:
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{W}</div><div class="lbl">Genişlik (px)</div></div>
  <div class="metric-box"><div class="val">{H}</div><div class="lbl">Yükseklik (px)</div></div>
</div>
<div class="metric-box" style="margin-top:0.5rem">
  <div class="val">{uploaded_file.size / 1024:.0f} KB</div>
  <div class="lbl">Dosya Boyutu</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀  ANALİZİ BAŞLAT", use_container_width=True)

    if run_btn:
        st.markdown("---")
        st.markdown("""
<div class="step-card">
  <div class="step-num">ADIM 02</div>
  <div class="step-title">İşleniyor…</div>
</div>
""", unsafe_allow_html=True)

        # Adım adım ilerleme
        progress = st.progress(0, text="Başlatılıyor…")
        log_area = st.empty()
        logs     = []

        def log(msg):
            logs.append(f"▸ {msg}")
            log_area.code("\n".join(logs), language=None)

        # 1 — Su maskesi
        progress.progress(10, text="Su bölgeleri tespit ediliyor…")
        log("Su tespiti başladı (HSV + konveks gövde filtresi)")
        water_mask = get_water_mask(img_rgb)
        water_pct  = (water_mask > 0).sum() / water_mask.size * 100
        log(f"Su alanı: %{water_pct:.1f}")
        progress.progress(22, text="Arazi sınıflandırılıyor…")

        # 2 — Arazi
        terrain_classes = classify_terrain(img_rgb, water_mask)
        log("Arazi sınıflandırması tamamlandı (bitki örtüsü, kıyı, kentsel)")
        progress.progress(34, text="AI modeli yükleniyor…")

        # 3 — Model
        log("MiDaS Small yükleniyor (önbellekte varsa hızlı)…")
        model, transform, device = load_ai_model()
        device_name = "GPU (CUDA)" if device.type == "cuda" else "CPU"
        log(f"Model hazır — Cihaz: {device_name}")
        progress.progress(46, text="Derinlik tahmini yapılıyor…")

        # 4 — DEM
        log("AI derinlik tahmini + DEM üretimi başladı…")
        dem, has_water = build_dem(img_rgb, model, transform, device, terrain_classes)
        log(f"DEM tamamlandı — Su bölgesi: {'VAR ✓' if has_water else 'YOK'}")
        progress.progress(72, text="Harita görselleştiriliyor…")

        # 5 — Debug gölge
        if debug_shadow:
            log("Gölge debug görünümü oluşturuluyor…")
            shadow_mask_dbg, _ = build_shadow_compensation(img_rgb)
            fig_dbg, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(img_rgb);                        axes[0].set_title("Orijinal");       axes[0].axis('off')
            sm = axes[1].imshow(shadow_mask_dbg, cmap='hot_r'); axes[1].set_title("Gölge Maskesi"); axes[1].axis('off')
            dm = axes[2].imshow(dem, cmap='gray');          axes[2].set_title("DEM (Gri)");     axes[2].axis('off')
            plt.colorbar(sm, ax=axes[1], fraction=0.035)
            plt.colorbar(dm, ax=axes[2], fraction=0.035)
            plt.tight_layout()
            buf_dbg = io.BytesIO()
            plt.savefig(buf_dbg, dpi=150, bbox_inches='tight')
            buf_dbg.seek(0)
            plt.close(fig_dbg)
            st.image(buf_dbg, caption="Gölge Debug Görünümü", use_container_width=True)

        # 6 — Ana görselleştirme
        log("Kontur + hillshade hesaplanıyor…")
        if save_3d:
            fig = plt.figure(figsize=(22, 8), facecolor='#f8f5f0')
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(img_rgb)
            ax1.set_title("Orijinal Uydu Görüntüsü", fontsize=12, fontweight='bold')
            ax1.axis('off')
            ax2 = fig.add_subplot(1, 3, 2)
            draw_contour_map(ax2, dem, has_water, contour_interval,
                             f"2D Topografik Harita\n(Kontur aralığı: {contour_interval:.3f})")
            draw_3d_surface(fig, dem, has_water, position=133)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#f8f5f0')
            ax1.imshow(img_rgb); ax1.set_title("Orijinal Uydu Görüntüsü"); ax1.axis('off')
            draw_contour_map(ax2, dem, has_water, contour_interval)

        plt.suptitle("Uydu Görüntüsünden Topografik Harita Üretimi  —  v7",
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        buf_main = io.BytesIO()
        plt.savefig(buf_main, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf_main.seek(0)
        main_img_bytes = buf_main.getvalue()
        plt.close(fig)
        log("Görselleştirme tamamlandı ✓")
        progress.progress(90, text="DEM dışa aktarılıyor…")

        # 7 — DEM PNG
        dem_u8        = (dem * 255).astype(np.uint8)
        dem_img_pil   = Image.fromarray(dem_u8)
        buf_dem       = io.BytesIO()
        dem_img_pil.save(buf_dem, format="PNG")
        dem_png_bytes = buf_dem.getvalue()

        # 8 — GeoTIFF (isteğe bağlı)
        geotiff_bytes = None
        if save_geotiff_flag:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tif_path = tmp.name
            ok = save_geotiff(dem, tif_path)
            if ok:
                with open(tif_path, "rb") as f:
                    geotiff_bytes = f.read()
                os.unlink(tif_path)
                log("GeoTIFF oluşturuldu ✓")
            else:
                log("GeoTIFF atlandı (rasterio yüklü değil)")

        progress.progress(100, text="Tamamlandı! ✓")
        log_area.empty()

        # ─────────────────────────────────────────────────────
        # SONUÇ GÖSTERİMİ
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
<div class="step-card">
  <div class="step-num">ADIM 03</div>
  <div class="step-title">Sonuçlar</div>
</div>
""", unsafe_allow_html=True)

        # Metrikler
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="val">%{water_pct:.1f}</div>
    <div class="lbl">Su Alanı</div>
  </div>
  <div class="metric-box">
    <div class="val">{'VAR' if has_water else 'YOK'}</div>
    <div class="lbl">Su Tespiti</div>
  </div>
  <div class="metric-box">
    <div class="val">{device_name.split()[0]}</div>
    <div class="lbl">İşlem Birimi</div>
  </div>
  <div class="metric-box">
    <div class="val">{contour_interval:.2f}</div>
    <div class="lbl">Kontur Aralığı</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Sekmeli görünüm
        tab1, tab2 = st.tabs(["📊  TOPOGRAFİK HARİTA", "🌑  DEM (GRİ)"])

        with tab1:
            st.image(main_img_bytes, caption="Topografik Harita — v7", use_column_width=True)

        with tab2:
            st.image(dem_png_bytes, caption="Yükseklik Haritası (Gri Ton)", use_column_width=True)
            st.caption("0 = Alçak (siyah) · 255 = Zirve (beyaz)")

        # İndirme butonları
        st.markdown("#### ⬇️ İndir")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        base = uploaded_file.name.rsplit(".", 1)[0]

        with dl_col1:
            st.download_button(
                label="📥  Topografik Harita (PNG)",
                data=main_img_bytes,
                file_name=f"{base}_topografik_harita_v7.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📥  DEM Gri Ton (PNG)",
                data=dem_png_bytes,
                file_name=f"{base}_DEM_gray.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl_col3:
            if geotiff_bytes:
                st.download_button(
                    label="📥  GeoTIFF",
                    data=geotiff_bytes,
                    file_name=f"{base}_DEM.tif",
                    mime="image/tiff",
                    use_container_width=True,
                )
            else:
                st.button("📥  GeoTIFF (rasterio gerekli)", disabled=True, use_container_width=True)

else:
    # Boş ekran talimatı
    st.markdown("""
<div style="text-align:center; padding: 4rem 0; color: #3a4f6a;">
  <div style="font-size: 3.5rem; margin-bottom: 1rem;">🛰️</div>
  <div style="font-family:'Space Mono',monospace; font-size:0.85rem; letter-spacing:2px; color:#2a7a5c;">
    UYDU FOTOĞRAFI YÜKLE → ANALİZİ BAŞLAT
  </div>
  <div style="font-size:0.8rem; color:#3a4f6a; margin-top:0.8rem;">
    PNG · JPG · TIFF formatları desteklenir
  </div>
</div>
""", unsafe_allow_html=True)