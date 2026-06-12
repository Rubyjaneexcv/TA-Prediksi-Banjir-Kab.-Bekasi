import streamlit as st
import pandas as pd
import joblib
import folium
import requests
import datetime
from streamlit_folium import st_folium

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Prediksi Banjir Kab. Bekasi",
    page_icon="🌊",
    layout="wide"
)

# --- 1. INISIALISASI SESSION STATE (Untuk Otomatisasi Slider) ---
if 'prec_h1' not in st.session_state: st.session_state.prec_h1 = 0.0
if 'gwettop_h0' not in st.session_state: st.session_state.gwettop_h0 = 0.5
if 'gwettop_h1' not in st.session_state: st.session_state.gwettop_h1 = 0.5
if 'gwetprof_h0' not in st.session_state: st.session_state.gwetprof_h0 = 0.5
if 'gwetprof_h1' not in st.session_state: st.session_state.gwetprof_h1 = 0.5

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

# --- 4. FUNGSI PENARIK DATA API (OPEN-METEO) ---
def fetch_weather_api(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&hourly=soil_moisture_0_to_7cm,soil_moisture_28_to_100cm&timezone=Asia%2FJakarta&past_days=1&forecast_days=1"
        response = requests.get(url)
        data = response.json()
        
        prec_kemarin = data['daily']['precipitation_sum'][0]
        sm_top_h1 = sum(data['hourly']['soil_moisture_0_to_7cm'][0:24]) / 24 * 2
        sm_top_h0 = sum(data['hourly']['soil_moisture_0_to_7cm'][24:48]) / 24 * 2
        sm_prof_h1 = sum(data['hourly']['soil_moisture_28_to_100cm'][0:24]) / 24 * 2
        sm_prof_h0 = sum(data['hourly']['soil_moisture_28_to_100cm'][24:48]) / 24 * 2
        
        return prec_kemarin, min(sm_top_h1, 1.0), min(sm_top_h0, 1.0), min(sm_prof_h1, 1.0), min(sm_prof_h0, 1.0)
    except Exception as e:
        return None

# --- 5. HEADER APLIKASI ---
st.title("🌊 Dashboard Spasial & Peringatan Dini Banjir")
st.subheader("Sistem Prediksi Berbasis Machine Learning - Kabupaten Bekasi")
st.markdown("---")

# --- 6. PEMBAGIAN LAYOUT ---
kolom_kiri, kolom_kanan = st.columns([6, 4])

with kolom_kiri:
    st.write("### 📍 Pilih Wilayah & Input Parameter Cuaca")
    pilihan_kec = st.selectbox("Pilih Kecamatan:", list(data_kecamatan.keys()))
    geo_data = data_kecamatan[pilihan_kec]
    
    st.info("💡 **Tips:** Klik tombol di bawah ini untuk menarik data cuaca aktual hari ini secara otomatis dari satelit.")
    if st.button("📡 Tarik Data Cuaca Otomatis (Live API)", use_container_width=True):
        with st.spinner('Menghubungkan ke satelit cuaca...'):
            hasil_api = fetch_weather_api(geo_data["lat"], geo_data["lon"])
            if hasil_api:
                st.session_state.prec_h1 = float(hasil_api[0])
                st.session_state.gwettop_h1 = float(hasil_api[1])
                st.session_state.gwettop_h0 = float(hasil_api[2])
                st.session_state.gwetprof_h1 = float(hasil_api[3])
                st.session_state.gwetprof_h0 = float(hasil_api[4])
                st.success(f"✅ Data cuaca terkini untuk {pilihan_kec} berhasil ditarik!")
            else:
                st.error("❌ Gagal menarik data. Periksa koneksi internetmu.")
    
    col_cuaca1, col_cuaca2 = st.columns(2)
    with col_cuaca1:
        st.number_input("Curah Hujan Kemarin (mm)", min_value=0.0, max_value=300.0, step=1.0, key='prec_h1')
        st.slider("Kelembaban Tanah Permukaan Hari Ini", 0.0, 1.0, key='gwettop_h0')
    with col_cuaca2:
        st.slider("Kelembaban Tanah Permukaan Kemarin", 0.0, 1.0, key='gwettop_h1')
        st.slider("Kelembaban Tanah Profil Hari Ini", 0.0, 1.0, key='gwetprof_h0')
        st.number_input("Kelembaban Tanah Profil Kemarin", 0.0, 1.0, key='gwetprof_h1', label_visibility="collapsed") 

    st.write("#### Visualisasi Lokasi Kecamatan:")
    peta = folium.Map(location=[geo_data["lat"], geo_data["lon"]], zoom_start=12)
    folium.Marker(
        [geo_data["lat"], geo_data["lon"]], popup=f"Kecamatan {pilihan_kec}", tooltip=pilihan_kec,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(peta)
    st_folium(peta, width="100%", height=300, key=f"map_{pilihan_kec}")

with kolom_kanan:
    st.write("### 📊 Analisis & Hasil Prediksi")
    st.info(f"**Karakteristik Fisik {pilihan_kec}:**\n* Rata-rata Elevasi: {geo_data['elevasi']} mdpl\n* Lahan Terbangun: {geo_data['built_up']*100}%\n* Luas Wilayah Risiko: {geo_data['luas_risiko']} Ha")
    
    if st.button("🔍 Jalankan Simulasi Prediksi", use_container_width=True):
        
        data_mentah = pd.DataFrame({
            'PRECTOTCORR': [0.0], 
            'GWETTOP': [st.session_state.gwettop_h0],
            'GWETPROF': [st.session_state.gwetprof_h0],
            'PRECTOTCORR_H_1': [st.session_state.prec_h1],
            'GWETTOP_H_1': [st.session_state.gwettop_h1],
            'GWETPROF_H_1': [st.session_state.gwetprof_h1]
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
        
        # MENDAPATKAN TANGGAL HARI INI (WIB)
        waktu_wib = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
        nama_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        nama_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        
        hari_ini = nama_hari[waktu_wib.weekday()]
        tanggal_format = f"{hari_ini}, {waktu_wib.day} {nama_bulan[waktu_wib.month - 1]} {waktu_wib.year}"
        
        st.markdown(f"#### **Status Peringatan Dini ({tanggal_format}):**")
        if hasil_prediksi == 1:
            st.error(f"🚨 **SIAGA: POTENSI BANJIR**")
            st.metric(label="Probabilitas Risiko", value=f"{probabilitas:.2f}%")
            st.warning("Kombinasi saturasi tanah dan intensitas hujan memicu limpasan air yang tinggi.")
        else:
            st.success(f"✅ **AMAN: MINIM POTENSI BANJIR**")
            st.metric(label="Probabilitas Risiko", value=f"{probabilitas:.2f}%")
            st.info("Kapasitas tanah masih mampu menyerap curah hujan masa lalu dengan aman.")
