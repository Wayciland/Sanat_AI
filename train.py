import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def train_model():
    data_dir = "dataset/train"
    batch_size = 4
    num_epochs = 15
    learning_rate = 0.0003

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    print(f"Tespit Edilen Siniflar: {dataset.class_to_idx}")
    print(f"Toplam Görsel Sayisi: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Sanat/Ayrıntı tespiti için EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    print("\nEğitim Başliyor (EfficientNet-B0)...")
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1:02d}/{num_epochs} - Loss: {epoch_loss:.4f} - Doğruluk (Acc): %{epoch_acc:.1f}")

    torch.save(model.state_dict(), "sanat_modeli.pth")
    print("\nEĞİTİM TAMAMLANDI! 'sanat_modeli.pth' kaydedildi.")

if __name__ == "__main__":
    train_model()