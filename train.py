import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def modeli_egit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim Cihazi: {device}")

    # Dijital çizimleri daha iyi anlayabilmesi için güçlendirilmiş veri artırma
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_yolu = "dataset"
    if not os.path.exists(dataset_yolu):
        print("Hata: 'dataset' klasörü bulunamadi!")
        return

    train_data = datasets.ImageFolder(root=dataset_yolu, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    # EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    print("\n=== ENIMERA LENS AĞIR İDMAN BAŞLIYOR ===")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {running_loss/len(train_loader):.4f} | Doğruluk: %{epoch_acc:.2f}")

    torch.save(model.state_dict(), "sanat_modeli.pth")
    print("\n[BAŞARILI] Ağir idman tamamlandi. Güçlendirilmiş 'sanat_modeli.pth' oluşturuldu!")

if __name__ == "__main__":
    modeli_egit()