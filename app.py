import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="AI vs Human Art Detector",
    page_icon="🎨",
    layout="wide"
)

# --- OZEL CSS (Tasarim ve Durum Isiklari) ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .status-dot {
        height: 12px;
        width: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .green-dot {
        background-color: #00ff66;
        box-shadow: 0 0 8px #00ff66;
    }
    .red-dot {
        background-color: #ff3333;
        box-shadow: 0 0 8px #ff3333;
    }
    .gray-dot {
        background-color: #888888;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL YUKLEME VE HAZIRLIK ---
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ResNet18 mimarisini tanimlama
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 2)
    )
    
    # Egitilmis agirlıkları yukleme
    try:
        import os
# ...
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "art_detector_model.pth")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Model yuklenirken hata olustu: {e}")
        return None, device

model, device = load_trained_model()

# Görsel Ön İşleme Transformatörü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- BASSLIK VE ARAYUZ ---
st.title("🎨 AI vs. Human Art Detector")
st.write("Yuklediginiz gorselin bir insan sanatci tarafindan mi cizildigini yoksa Yapay Zeka tarafindan mi uretildigini analiz edin.")
st.divider()

# Çift Sütunlu Düzen
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📁 Gorsel Yukleme")
    uploaded_file = st.file_uploader("Bir gorsel secin veya buraya surukleyin...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yuklenen Gorsel", use_container_width=True)
        analyze_btn = st.button("🔍 Gorseli Analiz Et", type="primary", use_container_width=True)
    else:
        analyze_btn = False

with col_right:
    st.subheader("📊 Analiz Sonuclari")
    
    with st.container(border=True):
        if analyze_btn and uploaded_file is not None:
            with st.spinner("Model taraniyor ve analiz ediliyor..."):
                time.sleep(0.5)
                
                # Model Tahmini
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    ai_score = probabilities[0][1].item()
                
                human_score = 1.0 - ai_score
                
                # Durum Isigi ve Karar Mantigi
                if ai_score >= 0.50:
                    status_html = '<p style="font-size:18px; font-weight:bold;"><span class="status-dot red-dot"></span>Tespit: YAPAY ZEKA URETIMI (AI)</p>'
                    st.markdown(status_html, unsafe_allow_html=True)
                    st.error(f"Bu gorsel %{ai_score * 100:.1f} ihtimalle AI tarafindan uretilmistir.")
                else:
                    status_html = '<p style="font-size:18px; font-weight:bold;"><span class="status-dot green-dot"></span>Tespit: INSAN YAPIMI ESER</p>'
                    st.markdown(status_html, unsafe_allow_html=True)
                    st.success(f"Bu gorsel %{human_score * 100:.1f} ihtimalle bir insan sanatciya aittir.")
                
                st.divider()
                
                # Güven Oranı Metrikleri
                st.write("**Tespit Guven Orani:**")
                progress_val = ai_score if ai_score >= 0.50 else human_score
                st.progress(progress_val)
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("AI Ihtimali", f"%{ai_score * 100:.1f}")
                col_m2.metric("Insan Ihtimali", f"%{human_score * 100:.1f}")
        else:
            # Bekleme Durumundaki Isik (Gri)
            st.markdown('<p style="font-weight:bold; color:#777;"><span class="status-dot gray-dot"></span>Analiz Bekleniyor...</p>', unsafe_allow_html=True)
            st.caption("Gorseli yukledikten sonra 'Gorseli Analiz Et' butonuna basarak sonuclari gorebilirsiniz.")