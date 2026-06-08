import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# Konfigurasi Tampilan Halaman
st.set_page_config(
    page_title="Dashboard Prediksi Banjir Kab. Bekasi",
    page_icon="🌊",
    layout="wide"
)

# 1. Memuat Model dan Scaler
@st.cache_resource
def load_components():
    rf_model = joblib.load('model_rf_banjir.pkl')
    scaler = joblib.load('scaler_banjir.pkl')
    return rf_model, scaler

rf_model, scaler = load_components()

# 2. Database Geospasial 23 Kecamatan Kabupaten Bekasi
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

# 3. Header Aplikasi
st.title("🌊 Dashboard Spasial & Peringatan Dini Banjir")
st.subheader("Sistem Prediksi Berbasis Machine Learning - Kabupaten Bekasi")
st.markdown("---")

# 4. MEMBAGI LAYOUT (Kiri: Input & Peta | Kanan: Hasil Prediksi)
kolom_kiri, kolom_kanan = st.columns([6, 4])

with kolom_kiri:
    st.write("### 📍 Pilih Wilayah & Input Parameter Cuaca")
    pilihan_kec = st.selectbox("Pilih Kecamatan:", list(data_kecamatan.keys()))
    geo_data = data_kecamatan[pilihan_kec]
    
    col_cuaca1, col_cuaca2 = st.columns(2)
    with col_cuaca1:
        prec_h1 = st.number_input("Curah Hujan Kemarin (mm)", min_value=0.0, max_value=300.0, value=25.0, step=1.0)
        gwettop_hari_ini = st.slider("Kelembaban Tanah Permukaan Hari Ini", 0.0, 1.0, 0.7)
    with col_cuaca2:
        gwettop_h1 = st.slider("Kelembaban Tanah Permukaan Kemarin", 0.0, 1.0, 0.6)
        gwetprof_hari_ini = st.slider("Kelembaban Tanah Profil Hari Ini", 0.0, 1.0, 0.6)
        gwetprof_h1 = 0.5 
        
    st.write("#### Visualisasi Lokasi Kecamatan:")
    peta = folium.Map(location=[geo_data["lat"], geo_data["lon"]], zoom_start=12)
    folium.Marker(
        [geo_data["lat"], geo_data["lon"]],
        popup=f"Kecamatan {pilihan_kec}",
        tooltip=pilihan_kec,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(peta)
    st_folium(peta, width="100%", height=300, key=f"map_{pilihan_kec}")

with kolom_kanan:
    st.write("### 📊 Analisis & Hasil Prediksi")
    st.info(f"**Karakteristik Fisik {pilihan_kec}:**\n* Rata-rata Elevasi: {geo_data['elevasi']} mdpl\n* Persentase Lahan Terbangun: {geo_data['built_up']*100}%\n* Luas Wilayah Risiko: {geo_data['luas_risiko']} Ha")
    
    if st.button("🔍 Jalankan Simulasi Prediksi", use_container_width=True):
        
        # a. Format Data untuk Scaler
        data_mentah = pd.DataFrame({
            'PRECTOTCORR': [0.0], 
            'GWETTOP': [gwettop_hari_ini],
            'GWETPROF': [gwetprof_hari_ini],
            'PRECTOTCORR_H_1': [prec_h1],
            'GWETTOP_H_1': [gwettop_h1],
            'GWETPROF_H_1': [gwetprof_h1]
        })
        
        # b. Normalisasi Data
        kolom_dinamis = ['PRECTOTCORR', 'GWETTOP', 'GWETPROF', 'PRECTOTCORR_H_1', 'GWETTOP_H_1', 'GWETPROF_H_1']
        data_scaled = data_mentah.copy()
        data_scaled[kolom_dinamis] = scaler.transform(data_mentah[kolom_dinamis])
        
        # c. Menyusun X_input dengan data geospasial
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
        
        # d. PENGAMAN: Urutkan kolom secara dinamis agar 100% sama persis dengan saat model dilatih
        X_input = X_input[rf_model.feature_names_in_]
        
        # e. Eksekusi Prediksi
        hasil_prediksi = rf_model.predict(X_input)[0]
        probabilitas = rf_model.predict_proba(X_input)[0][1] * 100
        
        # f. Output Layar
        st.markdown("#### **Status Peringatan Dini:**")
        if hasil_prediksi == 1:
            st.error(f"🚨 **SIAGA: POTENSI BANJIR**")
            st.metric(label="Probabilitas Risiko", value=f"{probabilitas:.2f}%")
            st.warning("Kombinasi saturasi tanah dan intensitas hujan memicu limpasan air yang tinggi.")
        else:
            st.success(f"✅ **AMAN: MINIM POTENSI BANJIR**")
            st.metric(label="Probabilitas Risiko", value=f"{probabilitas:.2f}%")
            st.info("Kapasitas tanah masih mampu menyerap curah hujan masa lalu dengan aman.")