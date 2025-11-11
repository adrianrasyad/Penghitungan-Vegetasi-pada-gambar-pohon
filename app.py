import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI STREAMLIT ---
# Mengatur layout default menjadi lebar untuk tampilan yang lebih baik
st.set_page_config(layout="wide", page_title="Deteksi Vegetasi (DIP) - HSV Segmentasi")

# --- FUNGSI UTILITY ---

def load_color_image(uploaded_file):
    """Membaca file yang diunggah sebagai citra Berwarna (BGR) dan melakukan resize."""
    if uploaded_file is None:
        return None
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Baca sebagai citra Berwarna (OpenCV default BGR)
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_color is None:
            st.error("Gagal memuat citra. Pastikan format file didukung atau citra tidak rusak.")
            return None
        
        # Resize citra jika terlalu besar, untuk performa dan UI
        rows, cols, _ = img_color.shape
        if rows > 800 or cols > 800:
            # Tetapkan ukuran maksimum 800x800
            scale = 800 / max(rows, cols)
            new_size = (int(cols * scale), int(rows * scale))
            img_color = cv2.resize(img_color, new_size, interpolation=cv2.INTER_AREA)

        return img_color
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses citra: {e}")
        return None

def segment_hsv(img_bgr, h_min, h_max, s_min, s_max, v_min, v_max):
    """Melakukan segmentasi warna pada citra BGR menggunakan rentang HSV yang diberikan."""
    # 1. Konversi BGR ke HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. Definisikan Batas Warna
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    
    # 3. Buat Masking
    mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
    
    # 4. Terapkan Masking ke Citra Asli (BGR)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    
    # Hitung Jumlah Piksel yang Terdetksi (dibagi 255 karena mask bernilai 255)
    pixel_count = np.sum(mask > 0) / 255
    
    return mask, result, pixel_count

# --- MODUL UTAMA APLIKASI ---

def main_app():
    # -----------------------------
    # 1. Judul dan Deskripsi
    # -----------------------------
    st.title("🛰️ Penghitungan Vegetasi Berbasis Citra Satelit")
    st.markdown("""
        Aplikasi ini menggunakan Segmentasi Ruang Warna **HSV (Hue, Saturation, Value)** untuk mengisolasi 
        dan menghitung piksel vegetasi (pohon) dari citra yang diunggah.
    """)
    st.markdown("---")

    # -----------------------------
    # 2. Upload Citra
    # -----------------------------
    col_upload, col_space = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Citra Satelit Berwarna (JPG, PNG)", 
            type=["jpg", "jpeg", "png"],
            help="Unggah gambar screenshot dari Google Maps atau sumber satelit lainnya."
        )
        
    if uploaded_file is None:
        st.info("⬆️ Silakan unggah citra satelit Anda untuk memulai analisis.")
        return

    img_bgr = load_color_image(uploaded_file)
    
    if img_bgr is None:
        return
        
    st.success("✅ Citra berhasil dimuat. Lanjutkan ke langkah 2 untuk mengontrol segmentasi.")
    st.markdown("---")
    
    # -----------------------------
    # 3. Kontrol HSV (Pindah ke Main Body untuk UX yang lebih baik)
    # -----------------------------
    st.header("🔬 Langkah 2: Kontrol Segmentasi Warna (HSV)")
    with st.expander("Atur Parameter HSV untuk Menargetkan Warna Hijau 🌿", expanded=True):
        st.markdown("Piksel hijau yang terdeteksi bergantung pada rentang nilai HSV yang Anda tetapkan. Rentang default disarankan untuk vegetasi.")
        
        col_h, col_s, col_v = st.columns(3)
        
        with col_h:
            st.subheader("Hue (Warna)")
            st.caption("Rentang 0-179. Hijau umumnya: 30-85.")
            h_min = st.slider("H Min", 0, 179, 30, key="h_min")
            h_max = st.slider("H Max", 0, 179, 85, key="h_max")
        
        with col_s:
            st.subheader("Saturation (Kejenuhan)")
            st.caption("Rentang 0-255. Memisahkan vegetasi dari abu-abu.")
            s_min = st.slider("S Min", 0, 255, 50, key="s_min")
            s_max = st.slider("S Max", 0, 255, 255, key="s_max")
            
        with col_v:
            st.subheader("Value (Kecerahan)")
            st.caption("Rentang 0-255. Memisahkan bayangan dan highlight.")
            v_min = st.slider("V Min", 0, 255, 50, key="v_min")
            v_max = st.slider("V Max", 0, 255, 255, key="v_max")

    # -----------------------------
    # 4. Proses dan Tampilkan Hasil
    # -----------------------------
    
    st.header("📊 Langkah 3: Hasil Analisis Citra")
    
    # Panggil fungsi segmentasi
    mask, result_bgr, pixel_count = segment_hsv(
        img_bgr, h_min, h_max, s_min, s_max, v_min, v_max
    )
    
    # Konversi BGR ke RGB untuk tampilan di Streamlit
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # Tampilkan Hasil dalam 3 Kolom
    col_img, col_mask, col_metrics = st.columns(3)

    # Kolom 1: Citra Asli
    with col_img:
        st.subheader("1️⃣ Citra Asli (Input)")
        st.image(img_rgb, caption=f"Dimensi: {img_bgr.shape[1]}x{img_bgr.shape[0]} piksel", use_container_width=True)

    # Kolom 2: Hasil Segmentasi
    with col_mask:
        st.subheader("2️⃣ Hasil Deteksi (Piksel Hijau)")
        st.image(result_rgb, caption="Hanya piksel dalam rentang HSV yang ditampilkan.", use_container_width=True)

    # Kolom 3: Metrik
    with col_metrics:
        st.subheader("3️⃣ Ringkasan Metrik")
        
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        percentage = (pixel_count / total_pixels) * 100
        
        # Kartu Metrik Utama
        st.metric(
            label="Total Piksel Vegetasi Terdeteksi (Area Hijau)", 
            value=f"{int(pixel_count):,}".replace(",", "."),
            delta_color="normal",
            help="Jumlah piksel yang berhasil diisolasi oleh filter HSV Anda."
        )
        
        st.metric(
            label="Persentase Area Vegetasi", 
            value=f"{percentage:.2f} %",
            delta=f"dari {total_pixels:,} total piksel",
            delta_color="off"
        )
        
        st.markdown("---")
        
        with st.expander("Detail Masking Biner", expanded=False):
            st.caption("Visualisasi Masker (Putih = Area Terdeteksi)")
            st.image(mask, caption="Masker Biner", use_container_width=True)
            st.info(
                f"**Rentang HSV yang Digunakan:** H: [{h_min}, {h_max}], S: [{s_min}, {s_max}], V: [{v_min}, {v_max}]"
            )

if __name__ == "__main__":
    main_app()