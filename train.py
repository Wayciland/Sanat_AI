import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def modeli_egit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim Cihazi: {device}")

    # Ağır İdman Dönüşümleri (Augmentation - Veriyi Zorlaştırma)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_yolu = "dataset"
    if not os.path.exists(dataset_yolu):
        print("HATA: 'dataset' klasörü bulunamadi. Önce dataset_builder.py çaliştirin!")
        return

    train_data = datasets.ImageFolder(root=dataset_yolu, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True) # Düşük batch size = Daha hassas öğrenme

    # Mimari: EfficientNet-B0 (Tesis Seviyesi Tespit Motoru)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), # Aşırı öğrenmeyi (Overfitting) engeller
        nn.Linear(num_ftrs, 2)
    )
    
    model = model.to(device)
    kriter = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4) # AdamW ile ince ayar

    print("\n=== ENIMERA LENS AĞIR İDMAN BAŞLIYOR ===")
    epochs = 10  # Derin öğrenme için 10 tekrar
    for epoch in range(epochs):
        model.train()
        toplam_loss = 0.0
        dogru_tahmin = 0
        toplam_ornek = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = kriter(outputs, labels)
            loss.backward()
            optimizer.step()
            
            toplam_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            dogru_tahmin += torch.sum(preds == labels.data)
            toplam_ornek += inputs.size(0)
            
        acc = dogru_tahmin.double() / toplam_ornek
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {toplam_loss/len(train_loader):.4f} | Doğruluk: %{acc*100:.2f}")

    # Ağır İdmanlı Modeli Kaydet
    torch.save(model.state_dict(), "sanat_modeli.pth")
    print("\n[BAŞARILI] Ağir idman tamamlandi. Güçlendirilmiş 'sanat_modeli.pth' oluşturuldu!")

if __name__ == "__main__":
    modeli_egit()