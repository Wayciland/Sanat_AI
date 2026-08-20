import streamlit as st
from PIL import Image

# Sayfa konfigürasyonunu geniş mod yapalım
st.set_page_config(page_title="Sanat AI - AI Detector", layout="wide")

st.title("🛡️ Sanat AI - Görsel Doğrulama Sistemi")
st.caption("Yüklenen görselin insan yapimi mi yoksa yapay zekâ üretimi mi olduğunu analiz eder.")

st.divider()

# Dosya yükleme alanı
uploaded_file = st.file_uploader("Bir görsel yükleyin...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Ekranı iki sütuna bölüyoruz: Sol Görsel, Sağ Analiz Sonucu
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("🖼️ Yüklenen Görsel")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("📊 Analiz Sonucu")
        
        with st.spinner("Görsel taraniyor ve modelden geçiriliyor..."):
            # --- MODEL TAHMİN BÖLÜMÜ ---
            # Buradaki 'ai_score' ve 'human_score' değerlerini kendi model çıktınla bağlayacaksın
            # Örnek temsil değerler:
            ai_score = 0.85  # Modelden gelen AI olasılığı
            human_score = 1.0 - ai_score
            
            # Ana Karar Kartı
            if ai_score >= 0.60:
                st.error("⚠️ Yapay Zekâ Üretimi (AI Generated)")
            elif ai_score <= 0.40:
                st.success("🎨 İnsan Yapimi Sanat Eseri (Human Art)")
            else:
                st.warning("❓ Şüpheli / Belirsiz Çizim")

            # Yüzdelik İlerleme Çubukları
            st.write("**AI İhtimali:**")
            st.progress(ai_score)
            st.write(f"%{ai_score * 100:.1f}")
            
            st.write("**İnsan Yapimi İhtimali:**")
            st.progress(human_score)
            st.write(f"%{human_score * 100:.1f}")
            
            # Detaylı Bilgi Kutusunu Gizle/Göster
            with st.expander("🛠️ Teknik Detaylar"):
                st.json({
                    "Dosya Adi": uploaded_file.name,
                    "Çözünürlük": f"{image.size[0]}x{image.size[1]}px",
                    "Format": image.format,
                    "Model": "EfficientNet-B0 (v1)"
                })