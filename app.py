import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Enimera Lens - AI Sanat Dedektörü",
    page_icon="🎨",
    layout="centered"
)

# Model Yükleme Fonksiyonu
@st.cache_resource
def modeli_yukle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 2)
    )
    
    try:
        model.load_state_dict(torch.load("sanat_modeli.pth", map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, device

# Görsel Ön İşleme
def gorsel_isleme(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Arayüz Başlığı ve Açıklama
st.title("🎨 Enimera Lens v1.0.1")
st.caption("Yapay Zeka Sanati ve İnsan Eseri Tespit Sistemi")

st.markdown("""
Sistemimiz, yüklenen eserin **yapay zeka** (Midjourney, Stable Diffusion vb.) tarafindan mi üretildiğini yoksa **insan sanati** mi olduğunu analiz eder.
""")

st.divider()

# Yasal Onay Kutusu
onay = st.checkbox("Yüklediğim görselin analiz edilmesini ve sistem koşullarini onayliyorum.")

gorsel = st.file_uploader("Bir görsel yükleyin...", type=["jpg", "jpeg", "png", "webp"])

if gorsel and onay:
    image = Image.open(gorsel).convert("RGB")
    st.image(image, caption="Yüklenen Eser", use_container_width=True)
    
    with st.spinner("Enimera Lens piksel katmanlarini analiz ediyor..."):
        model, device = modeli_yukle()
        if model is not None:
            input_tensor = gorsel_isleme(image).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                ai_ihtimal = probabilities[0][0].item() * 100
                insan_ihtimal = probabilities[0][1].item() * 100

            st.divider()
            st.subheader("📊 Analiz Sonucu")
            
            col1, col2 = st.columns(2)
            col1.metric("Yapay Zeka (AI)", f"%{ai_ihtimal:.1f}")
            col2.metric("İnsan Sanati", f"%{insan_ihtimal:.1f}")
            
            if ai_ihtimal >= 50:
                st.error(f"🔴 Bu görsel **%{ai_ihtimal:.1f}** ihtimalle Yapay Zeka üretimidir.")
            else:
                st.success(f"🟢 Bu görsel **%{insan_ihtimal:.1f}** ihtimalle bir İnsan Eseridir.")

elif gorsel and not onay:
    st.warning("⚠️ Analizi başlatmak için lütfen yukaridaki onay kutusunu işaretleyin.")

st.divider()
st.caption("Enimera Art Fund © 2026 - Tüm Haklari Saklidir.")