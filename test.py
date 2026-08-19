import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = models.efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )

    model.load_state_dict(torch.load("sanat_modeli.pth", map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    class_names = ['Fully_Human (İnsan Çizimi)', 'Pure_AI (Yapay Zekâ)']

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    print(f"\n==============================")
    print(f"Görsel: {image_path}")
    print(f"Sonuç:  {class_names[predicted_idx.item()]}")
    print(f"Güven:  %{confidence.item() * 100:.2f}")
    print(f"==============================\n")

if __name__ == "__main__":
    # Eğer terminalden bir dosya adı verilmişse onu al, verilmediyse varsayılan resim1.png kullan
    if len(sys.argv) > 1:
        test_resim_yolu = sys.argv[1]
    else:
        test_resim_yolu = "resim1.png"
    
    if os.path.exists(test_resim_yolu):
        predict_image(test_resim_yolu)
    else:
        print(f"Hata: '{test_resim_yolu}' dosyasi bulunamadi!")