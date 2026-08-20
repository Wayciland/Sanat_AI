import streamlit as st
import time
import uuid
from PIL import Image

st.set_page_config(page_title="Sanat AI - Doğrulama Paneli", layout="wide")

# CSS ile özel kırmızı/yeşil nokta ışığı stilleri
st.markdown("""
    <style>
    .status-dot {
        height: 18px;
        width: 18px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .red-dot {
        background-color: #ff4b4b;
        box-shadow: 0 0 10px #ff4b4b;
    }
    .green-dot {
        background-color: #00c853;
        box-shadow: 0 0 10px #00c853;
    }
    .gray-dot {
        background-color: #888888;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Sanat AI - Görsel Doğrulama Sistemi")

# Tasaıma Uygun 2 Ana Sütun Düzeni
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    # 1. Görseli Sürükle / Yükle Alanı
    st.subheader("🖼️ Görseli Sürükle")
    uploaded_file = st.file_uploader(
        "Görselinizi buraya sürükleyin veya bilgisayardan seçin", 
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=f"Yüklenen: {uploaded_file.name}")
        
        # 2. Görseli Analiz Et Butonu
        analyze_btn = st.button("⚡ Görseli Analiz Et", type="primary", use_container_width=True)
    else:
        analyze_btn = False

with col_right:
    # 3. Detaylar Kutusu
    st.subheader("📋 Detaylar")
    with st.container(border=True):
        if uploaded_file is not None:
            # Otomatik Benzersiz Analiz ID'si ve Detaylar
            analysis_id = str(uuid.uuid4())[:8].upper()
            st.markdown(f"**Eser Analiz ID:** `SAI-{analysis_id}`")
            st.markdown(f"**Proje Adi:** Sanat AI Detector v1")
            st.markdown(f"**Dosya Adi:** `{uploaded_file.name}`")
            st.markdown(f"**Çözünürlük:** {image.size[0]} x {image.size[1]} px")
            st.markdown(f"**Format / Boyut:** {image.format} ({round(uploaded_file.size / 1024, 1)} KB)")
        else:
            st.info("Görsel yüklendiğinde eser ID'si ve teknik detaylar burada görünecektir.")

    st.write("") # Dikey boşluk
    
    # 4. Sonuçlar ve Durum Işıkları Kutusu
    st.subheader("📊 Sonuçlar")
    
    with st.container(border=True):
        if analyze_btn and uploaded_file is not None:
            with st.spinner("Model taraniyor ve analiz ediliyor..."):
                time.sleep(1) # Şık bir yükleme efekti
                
                # --- MODEL TAHMİNİ (Kendi model çıktın ile burayı bağlayabilirsin) ---
                ai_score = 0.88  # Örnek AI skoru (%88)
                human_score = 1.0 - ai_score
                
                # Durum Işığı ve Karar Mantığı
                if ai_score >= 0.50:
                    status_html = '<p style="font-size:18px; font-weight:bold;"><span class="status-dot red-dot"></span>Tespit: YAPAY ZEKÂ ÜRETİMİ (AI)</p>'
                    st.markdown(status_html, unsafe_allow_html=True)
                    st.error(f"Bu görsel %{ai_score * 100:.1f} ihtimalle AI tarafindan üretilmiştir.")
                else:
                    status_html = '<p style="font-size:18px; font-weight:bold;"><span class="status-dot green-dot"></span>Tespit: İNSAN YAPIMI ESER</p>'
                    st.markdown(status_html, unsafe_allow_html=True)
                    st.success(f"Bu görsel %{human_score * 100:.1f} ihtimalle bir insan sanatçiya aittir.")
                
                st.divider()
                
                # Güven Oranı Metrikleri
                st.write("**Tespit Güven Orani:**")
                st.progress(ai_score if ai_score >= 0.50 else human_score)
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("AI İhtimali", f"%{ai_score * 100:.1f}")
                col_m2.metric("İnsan İhtimali", f"%{human_score * 100:.1f}")

        else:
            # Bekleme Durumundaki Işık (Gri)
            st.markdown('<p style="font-weight:bold; color:#777;"><span class="status-dot gray-dot"></span>Analiz Bekleniyor...</p>', unsafe_allow_html=True)
            st.caption("Görseli yükledikten sonra 'Görseli Analiz Et' butonuna basarak sonuçlari görebilirsiniz.")