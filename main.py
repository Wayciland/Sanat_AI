import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import torch.nn as nn
from torchvision import transforms, models
import gradio as gr
import c2pa
from serpapi import GoogleSearch

# ---------------------------------------------------------
# 1. MODEL YAPISI VE YÜKLEME (2 SINIFLI)
# ---------------------------------------------------------
class ArtOriginClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ArtOriginClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

SINIFLAR = ["Fully_Human (Insan Yapimi)", "Pure_AI (Tamamen Yapay Zeka)"]
model = ArtOriginClassifier(num_classes=len(SINIFLAR))

if os.path.exists("sanat_modeli.pth"):
    model.load_state_dict(torch.load("sanat_modeli.pth", map_location=torch.device('cpu')))
    print("BAŞARILI: Eğitilmiş özel model akilli olarak sisteme yüklendi!")
else:
    print("BİLGİ: 'sanat_modeli.pth' bulunamadi. Şu an ham (eğitilmemiş) model çalişiyor.")

model.eval()

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 2. ANALİZ FONKSİYONLARI
# ---------------------------------------------------------

# FFT Analizi
def analyze_art_fft(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    mask_size = 30
    magnitude_high = np.copy(np.abs(f_shift))
    magnitude_high[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0

    high_freq_energy = np.mean(magnitude_high)
    total_energy = np.mean(np.abs(f_shift))
    high_freq_ratio = float(high_freq_energy / total_energy)

    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='magma')
    plt.title('FFT Spektrum Haritasi')
    plt.axis('off')
    spectrum_path = 'temp_spectrum.png'
    plt.savefig(spectrum_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return high_freq_ratio, spectrum_path

# EXIF Analizi
def check_exif_data(pil_img):
    exif_info = {}
    ai_keywords = ['midjourney', 'dall-e', 'stable diffusion', 'comfyui', 'novelai', 'automatic1111']
    software_found = "Tespit Edilemedi"
    ai_trace_detected = False

    try:
        info = pil_img._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif_info[decoded] = value
                
                if decoded in ['Software', 'ProcessingSoftware', 'Make', 'Model']:
                    val_str = str(value).lower()
                    software_found = str(value)
                    
                    if any(keyword in val_str for keyword in ai_keywords):
                        ai_trace_detected = True

        raw_info = str(pil_img.info).lower()
        if any(keyword in raw_info for keyword in ai_keywords):
            ai_trace_detected = True
            software_found = "Yapay Zeka Metadata İzi (PNG/JPEG Header)"

    except Exception:
        pass

    return software_found, ai_trace_detected

# C2PA Mühür Analizi
def check_c2pa_manifest(image_path):
    try:
        reader = c2pa.Reader.from_file(image_path)
        manifest_json = reader.json()
        
        if "ai" in manifest_json.lower() or "generator" in manifest_json.lower():
            return "EVET (C2PA Dijital AI Mührü Doğrulandi!)"
        return "EVET (Geçerli C2PA İmzasi Var - İnsan Sanatçi/Cihaz)"
    except Exception:
        return "HAYIR (C2PA Mührü Bulunamadi)"

# Tersine Görsel Arama (SerpAPI)
def perform_reverse_image_search(image_path):
    api_key = "SERPAPI_KEY_BURAYA_GELECEK" 
    
    if api_key == "SERPAPI_KEY_BURAYA_GELECEK":
        return "Arama Yapilamadi (API Key Tanimlanmadi)"

    try:
        params = {
            "engine": "google_reverse_image",
            "image_url": image_path,
            "api_key": api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "inline_images" in results:
            return f"EVET ({len(results['inline_images'])} Benzer Görsel Bulundu)"
        return "HAYIR (İnternette Birebir Eşleşme Bulunamadi)"
    except Exception as e:
        return f"Arama Yapılamadı: {str(e)}"

# Ana İşleme Fonksiyonu
def process_image(input_image):
    if input_image is None:
        return None, "Görsel Yüklenmedi", {}
    
    pil_img = Image.fromarray(input_image).convert('RGB')
    
    temp_path = "temp_analysis_img.png"
    pil_img.save(temp_path)

    yazilim_bilgisi, exif_ai_suphesi = check_exif_data(pil_img)
    c2pa_durum = check_c2pa_manifest(temp_path)
    tersine_arama_durum = perform_reverse_image_search(temp_path)
    fft_skoru, spectrum_img_path = analyze_art_fft(input_image)
    
    tensor_img = transform_pipeline(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    confidences = {SINIFLAR[i]: float(probabilities[i]) for i in range(len(SINIFLAR))}
    
    max_prob = max(probabilities).item()
    max_index = torch.argmax(probabilities).item()
    tahmin_sinifi = SINIFLAR[max_index]

    fft_suphe = "EVET" if fft_skoru > 0.88 else "HAYIR"
    exif_suphe = "EVET (Yapay Zeka Yazilim İzi Bulundu!)" if exif_ai_suphesi else "HAYIR"

    if max_prob < 0.60:
        nihai_karar = "KARARSIZ / MANUEL İNCELEME GEREKLİ (Model Güven Orani Yetersiz)"
    else:
        nihai_karar = f"{tahmin_sinifi} (Güven: %{max_prob*100:.2f})"

    fft_rapor = (
        f"Kullanilan/Algilanan Yazilim : {yazilim_bilgisi}\n"
        f"EXIF Yapay Zeka İzi Şüphesi : {exif_suphe}\n"
        f"C2PA Dijital Mühür Durumu  : {c2pa_durum}\n"
        f"Tersine Görsel Arama Tespiti: {tersine_arama_durum}\n"
        f"----------------------------------------\n"
        f"FFT Anomali Skoru          : {fft_skoru:.4f}\n"
        f"Frekans Yapay Zeka İzi      : {fft_suphe}\n"
        f"----------------------------------------\n"
        f"Nihai Model Değerlendirmesi : {nihai_karar}"
    )
    
    return spectrum_img_path, fft_rapor, confidences

# ---------------------------------------------------------
# 3. GRADIO ARAYÜZÜ VE HUKUKİ ONAY
# ---------------------------------------------------------
with gr.Blocks(title="Sanat Eseri AI Orijinallik Tespit Sistemi") as demo:
    gr.Markdown("# Sanat Eseri AI Orijinallik Tespit Sistemi")
    gr.Markdown("Görseli analiz eder. Model güven orani %60'in altinda kaldiğinda otomatik olarak 'Manuel İnceleme' uyarisi verir.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Eseri Yükleyin")
            
            legal_checkbox = gr.Checkbox(
                label="Kullanim Şartlari ve KVKK Metnini Okudum, Kabul Ediyorum", 
                value=False
            )
            
            with gr.Accordion("Hukuki Sorumluluk ve Gizlilik Metni (Açmak için tiklayin)", open=False):
                gr.Markdown("""
                **KULLANIM ŞARTLARI VE GİZLİLİK BİLDİRİMİ**
                
                1. **Sorumluluk Reddi:** Bu platform tarafindan sunulan analiz sonuçlari olasilik bazli tahminlerdir ve %100 kesinlik ifade etmez. Sistem bir nihai yargi mekanizmasi değil, karar destek aracidir.
                2. **Telif Haklari:** Yüklediğiniz görselin analiz edilmesine yetkili olduğunuzu kabul etmektesiniz.
                3. **Veri Gizliliği:** Yüklenen görseller yalnizca analiz süresince geçici olarak işlenir. KVKK uyarinca izinsiz saklanmaz veya üçüncü şahislarla paylaşilmaz.
                """)
            
            submit_btn = gr.Button("Analiz Et", variant="primary")
            
        with gr.Column():
            output_spectrum = gr.Image(label="FFT Spektrum Haritasi")
            output_report = gr.Textbox(label="Sistem İnceleme Raporu")
            output_label = gr.Label(label="Sinif Dağilim Oranlari")

    def check_and_process(image, legal_accepted):
        if not legal_accepted:
            return None, "HATA: Lütfen analize başlamadan önce Hukuki Şartlari ve KVKK Metnini onaylayin.", {}
        return process_image(image)

    submit_btn.click(
        fn=check_and_process,
        inputs=[input_img, legal_checkbox],
        outputs=[output_spectrum, output_report, output_label]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)