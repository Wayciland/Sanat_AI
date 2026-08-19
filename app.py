import streamlit as st
st.set_page_config(page_title="Sanat AI", layout="wide") 
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
st.title("🎨 Sanat AI - Yapay Zekâ Tespit Paneli")
st.write("Yüklenen görselin bir **İnsan Çizimi** mi yoksa **Yapay Zekâ üretimi** mi olduğunu analiz eder.")

# Cihaz Ayarı ve Model Yükleme
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "sanat_modeli.pth"
    
    if not os.path.exists(model_path):
        return None, device

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

if model is None:
    st.error(" Model dosyasi (`sanat_modeli.pth`) bulunamadi!")
else:
    # Görsel Yükleme Alanı
    uploaded_file = st.file_uploader("Bir görsel seçin veya sürükleyin...", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

        if st.button(" Görseli Analiz Et", type="primary"):
            with st.spinner("Model analiz yapiyor..."):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                input_tensor = transform(image).unsqueeze(0).to(device)
                class_names = ['Fully_Human (İnsan Çizimi)', 'Pure_AI (Yapay Zekâ)']

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)

                res_text = class_names[predicted_idx.item()]
                conf_val = confidence.item() * 100

                st.divider()
                if predicted_idx.item() == 0:
                    st.success(f"**Sonuç:** {res_text}")
                else:
                    st.error(f"**Sonuç:** {res_text}")

                st.info(f"**Güven Orani:** %{conf_val:.2f}")