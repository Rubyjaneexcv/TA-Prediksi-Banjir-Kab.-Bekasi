import streamlit as st
import pandas as pd
import joblib
import folium
import requests
import datetime
from streamlit_folium import st_folium

# Set konfigurasi halaman agar tampilan lebih luas
st.set_page_config(page_title="Dashboard Banjir Bekasi", layout="wide")

# --- 1. SUNTIK CSS KUSTOM (Mengecilkan Ukuran Header) ---
st.markdown(
    """
    <style>
    .judul-kustom {
        font-size: 26px !important; 
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
        margin-top: -40px; /* Menarik konten lebih ke atas */
    }
    .sub-judul-kustom {
        font-size: 16px !important;
        color: #34495e;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render Header yang sudah dikecilkan ukurannya
st.markdown('<p class="judul-kustom">🌊 Dashboard Spasial & Peringatan Dini Banjir</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-judul-kustom">Sistem Prediksi Berbasis Machine Learning - Kabupaten Bekasi</p>', unsafe_allow_html=True)
st.markdown("---")
# --- 1. INISIALISASI SESSION STATE ---
if 'api_forecast' not in st.session_state: st.session_state.api_forecast = None
if 'mode_manual' not in st.session_state: st.session_state.mode_manual = False

# --- 2. MEMUAT MODEL DAN SCALER ---
@st.cache_resource
def load_components():
    rf_model = joblib.load('model_rf_banjir.pkl')
    scaler = joblib.load('scaler_banjir.pkl')
    return rf_model, scaler

rf_model, scaler = load_components()

# --- 3. DATABASE GEOSPASIAL 23 KECAMATAN ---
data_kecamatan = {
    "Babelan": {"lat": -6.1622, "lon": 107.0075, "elevasi": 5.0, "built_up": 0.55, "luas_risiko": 340.8, "jiwa_terpapar": 12000},
    "Bojongmangu": {"lat": -6.4440, "lon": 107.1640, "elevasi": 50.0, "built_up": 0.30, "luas_risiko": 50.0, "jiwa_terpapar": 2000},
    "Cabangbungin": {"lat": -6.0460, "lon": 107.1590, "elevasi": 3.0, "built_up": 0.20, "luas_risiko": 150.0, "jiwa_terpapar": 3000},
    "Cibarusah": {"lat": -6.4250, "lon": 107.1260, "elevasi": 60.0, "built_up": 0.35, "luas_risiko": 80.0, "jiwa_terpapar": 4000},
    "Cibitung": {"lat": -6.2514, "lon": 107.1033, "elevasi": 15.0, "built_up": 0.65, "luas_risiko": 150.3, "jiwa_terpapar": 8000},
    "Cikarang Barat": {"lat": -6.2690, "lon": 107.1020, "elevasi": 20.0, "built_up": 0.70, "luas_risiko": 200.0, "jiwa_terpapar": 10000},
    "Cikarang Pusat": {"lat": -6.3644, "lon": 107.1725, "elevasi": 25.0, "built_up": 0.45, "luas_risiko": 120.5, "jiwa_terpapar": 5000},
    "Cikarang Selatan": {"lat": -6.3260, "lon": 107.1350, "elevasi": 30.0, "built_up": 0.60, "luas_risiko": 180.0, "jiwa_terpapar": 9000},
    "Cikarang Timur": {"lat": -6.2850, "lon": 107.1900, "elevasi": 22.0, "built_up": 0.50, "luas_risiko": 160.0, "jiwa_terpapar": 7000},
    "Cikarang Utara": {"lat": -6.2570, "lon": 107.1500, "elevasi": 18.0, "built_up": 0.65, "luas_risiko": 190.0, "jiwa_terpapar": 8500},
    "Karangbahagia": {"lat": -6.2160, "lon": 107.1680, "elevasi": 15.0, "built_up": 0.40, "luas_risiko": 110.0, "jiwa_terpapar": 4500},
    "Kedungwaringin": {"lat": -6.2600, "lon": 107.2530, "elevasi": 12.0, "built_up": 0.35, "luas_risiko": 90.0, "jiwa_terpapar": 3500},
    "Muaragembong": {"lat": -5.9920, "lon": 107.0150, "elevasi": 1.0, "built_up": 0.10, "luas_risiko": 500.0, "jiwa_terpapar": 15000},
    "Pebayuran": {"lat": -6.1400, "lon": 107.2340, "elevasi": 8.0, "built_up": 0.25, "luas_risiko": 130.0, "jiwa_terpapar": 4000},
    "Serang Baru": {"lat": -6.3810, "lon": 107.1000, "elevasi": 45.0, "built_up": 0.40, "luas_risiko": 100.0, "jiwa_terpapar": 5000},
    "Setu": {"lat": -6.3350, "lon": 107.0390, "elevasi": 35.0, "built_up": 0.45, "luas_risiko": 140.0, "jiwa_terpapar": 6000},
    "Sukakarya": {"lat": -6.1550, "lon": 107.1630, "elevasi": 10.0, "built_up": 0.30, "luas_risiko": 95.0, "jiwa_terpapar": 3000},
    "Sukatani": {"lat": -6.1760, "lon": 107.1750, "elevasi": 12.0, "built_up": 0.35, "luas_risiko": 105.0, "jiwa_terpapar": 3500},
    "Sukawangi": {"lat": -6.1130, "lon": 107.1100, "elevasi": 5.0, "built_up": 0.25, "luas_risiko": 120.0, "jiwa_terpapar": 4000},
    "Tambelang": {"lat": -6.1630, "lon": 107.1130, "elevasi": 10.0, "built_up": 0.30, "luas_risiko": 85.0, "jiwa_terpapar": 2500},
    "Tambun Selatan": {"lat": -6.2641, "lon": 107.0614, "elevasi": 18.0, "built_up": 0.75, "luas_risiko": 210.2, "jiwa_terpapar": 15000},
    "Tambun Utara": {"lat": -6.2080, "lon": 107.0540, "elevasi": 10.0, "built_up": 0.60, "luas_risiko": 170.0, "jiwa_terpapar": 9000},
    "Tarumajaya": {"lat": -6.1153, "lon": 106.9881, "elevasi": 2.0, "built_up": 0.40, "luas_risiko": 410.1, "jiwa_terpapar": 9500}
}

# --- 4. FUNGSI API 7 HARI KE DEPAN (One-Week Forecast) ---
def fetch_weather_forecast(lat, lon):
    try:
        # Menarik data H-1 (past_days=1) dan 7 hari ke depan (forecast_days=7)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&hourly=soil_moisture_0_to_7cm,soil_moisture_28_to_100cm&timezone=Asia%2FJakarta&past_days=1&forecast_days=7"
        response = requests.get(url)
        data = response.json()
        
        forecast_list = []
        
        # Looping untuk 7 Hari (Indeks 1 sampai 7)
        for i in range(1, 8):
            prec_h1 = data['daily']['precipitation_sum'][i-1] 
            
            sm_top_h1 = sum(data['hourly']['soil_moisture_0_to_7cm'][(i-1)*24 : i*24]) / 24 * 2
            sm_top_h0 = sum(data['hourly']['soil_moisture_0_to_7cm'][i*24 : (i+1)*24]) / 24 * 2
            sm_prof_h1 = sum(data['hourly']['soil_moisture_28_to_100cm'][(i-1)*24 : i*24]) / 24 * 2
            sm_prof_h0 = sum(data['hourly']['soil_moisture_28_to_100cm'][i*24 : (i+1)*24]) / 24 * 2
            
            forecast_list.append({
                'hari_ke': i,
                'prec_h1': prec_h1,
                'gwettop_h1': min(sm_top_h1, 1.0),
                'gwettop_h0': min(sm_top_h0, 1.0),
                'gwetprof_h1': min(sm_prof_h1, 1.0),
                'gwetprof_h0': min(sm_prof_h0, 1.0)
            })
        return forecast_list
    except Exception as e:
        return None

# --- FUNGSI PREDIKSI UTAMA ---
def jalankan_prediksi(data_cuaca_dict, geo_data):
    data_mentah = pd.DataFrame({
        'PRECTOTCORR': [0.0], 
        'GWETTOP': [data_cuaca_dict['gwettop_h0']],
        'GWETPROF': [data_cuaca_dict['gwetprof_h0']],
        'PRECTOTCORR_H_1': [data_cuaca_dict['prec_h1']],
        'GWETTOP_H_1': [data_cuaca_dict['gwettop_h1']],
        'GWETPROF_H_1': [data_cuaca_dict['gwetprof_h1']]
    })
    
    kolom_dinamis = ['PRECTOTCORR', 'GWETTOP', 'GWETPROF', 'PRECTOTCORR_H_1', 'GWETTOP_H_1', 'GWETPROF_H_1']
    data_scaled = data_mentah.copy()
    data_scaled[kolom_dinamis] = scaler.transform(data_mentah[kolom_dinamis])
    
    X_input = pd.DataFrame({
        'GWETTOP': data_scaled['GWETTOP'],
        'GWETPROF': data_scaled['GWETPROF'],
        'PRECTOTCORR_H_1': data_scaled['PRECTOTCORR_H_1'],
        'GWETTOP_H_1': data_scaled['GWETTOP_H_1'],
        'GWETPROF_H_1': data_scaled['GWETPROF_H_1'],
        'AVG_Luas_Risiko': [geo_data['luas_risiko']],
        'AVG_Jiwa_Terpapar': [geo_data['jiwa_terpapar']],
        'AVG_ELEVATION': [geo_data['elevasi']],
        'PERC_BUILT_UP': [geo_data['built_up']]
    })
    
    X_input = X_input[rf_model.feature_names_in_]
    hasil_prediksi = rf_model.predict(X_input)[0]
    probabilitas = rf_model.predict_proba(X_input)[0][1] * 100
    
    return hasil_prediksi, probabilitas

# --- 5. HEADER APLIKASI ---
st.title("🌊 Dashboard Spasial & Peringatan Dini Banjir")
st.subheader("Sistem Prediksi Berbasis Machine Learning - Kabupaten Bekasi")
st.markdown("---")

# --- 6. PEMBAGIAN LAYOUT ---
kolom_kiri, kolom_kanan = st.columns([4, 6])

with kolom_kiri:
    st.write("### 📍 Pilih Wilayah Kabupaten Bekasi")
    pilihan_kec = st.selectbox("Pilih Kecamatan:", list(data_kecamatan.keys()))
    geo_data = data_kecamatan[pilihan_kec]
    
    # TOMBOL UTAMA (FITUR 7 HARI)
    st.info("💡 **Rekomendasi:** Klik tombol di bawah untuk menarik data satelit dan memprediksi status banjir selama 1 minggu penuh ke depan.")
    if st.button("📡 Tarik & Prediksi Cuaca 7 Hari (Otomatis)", use_container_width=True):
        with st.spinner('Menghubungkan ke satelit cuaca Open-Meteo...'):
            st.session_state.api_forecast = fetch_weather_forecast(geo_data["lat"], geo_data["lon"])
            st.session_state.mode_manual = False
            if st.session_state.api_forecast:
                st.success(f"✅ Prakiraan cuaca 7 hari untuk {pilihan_kec} berhasil diproses!")
            else:
                st.error("❌ Gagal menarik data. Periksa koneksi internetmu.")
    
    st.write("#### Visualisasi Lokasi Kecamatan:")
    peta = folium.Map(location=[geo_data["lat"], geo_data["lon"]], zoom_start=12)
    folium.Marker(
        [geo_data["lat"], geo_data["lon"]], popup=f"Kecamatan {pilihan_kec}", tooltip=pilihan_kec,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(peta)
    st_folium(peta, width="100%", height=300, key=f"map_{pilihan_kec}")

with kolom_kanan:
    st.write("### 📊 Analisis & Hasil Prediksi (7 Hari)")
    st.info(f"**Karakteristik Fisik {pilihan_kec}:** Elevasi {geo_data['elevasi']} mdpl | Lahan Terbangun {geo_data['built_up']*100}% | Wilayah Risiko {geo_data['luas_risiko']} Ha")
    
    # JIKA TOMBOL API SUDAH DITEKAN, MUNCULKAN TAB 7 HARI!
    if st.session_state.api_forecast is not None and not st.session_state.mode_manual:
        waktu_wib = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
        nama_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        nama_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        
        # Membuat 7 Tab yang membentang rapi
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "HARI INI", "BESOK", "LUSA", "HARI KE-4", "HARI KE-5", "HARI KE-6", "HARI KE-7"
        ])
        tabs = [tab1, tab2, tab3, tab4, tab5, tab6, tab7]
        
        for idx, (tab, data_cuaca) in enumerate(zip(tabs, st.session_state.api_forecast)):
            with tab:
                # Menghitung tanggal kalender
                waktu_target = waktu_wib + datetime.timedelta(days=idx)
                hari_target = nama_hari[waktu_target.weekday()]
                tgl_format = f"{hari_target}, {waktu_target.day} {nama_bulan[waktu_target.month - 1]} {waktu_target.year}"
                
                st.write(f"**🗓️ Prakiraan: {tgl_format}**")
                
                # Menampilkan mini-dashboard parameter cuaca
                col_a, col_b = st.columns(2)
                col_a.metric("Curah Hujan Pemicu (H-1)", f"{data_cuaca['prec_h1']:.1f} mm")
                col_b.metric("Saturasi Tanah Terprediksi", f"{data_cuaca['gwettop_h0']:.2f}")
                
                # Memasukkan angka cuaca ke dalam mesin Machine Learning
                hasil, prob = jalankan_prediksi(data_cuaca, geo_data)
                
                st.markdown("#### **Status Peringatan Dini:**")
                if hasil == 1:
                    st.error(f"🚨 **SIAGA: POTENSI BANJIR**")
                    st.metric(label="Probabilitas Risiko Terjadi", value=f"{prob:.2f}%")
                    st.warning("Kombinasi saturasi tanah dan prakiraan intensitas hujan berpotensi memicu limpasan air tinggi.")
                else:
                    st.success(f"✅ **AMAN: MINIM POTENSI BANJIR**")
                    st.metric(label="Probabilitas Risiko Terjadi", value=f"{prob:.2f}%")
                    st.info("Kondisi tanah diperkirakan masih mampu menyerap curah hujan dengan aman pada hari ini.")
                    
    else:
        st.write("👈 *Silakan klik tombol 'Tarik & Prediksi Cuaca 7 Hari (Otomatis)' di panel sebelah kiri untuk melihat prakiraan banjir 1 minggu penuh ke depan.*")
