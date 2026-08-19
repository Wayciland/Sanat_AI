import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

class ArtClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sanat AI - Yapay Zekâ Tespit Paneli")
        self.root.geometry("480x580")  # Boşluklar kaldırıldı
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        # Cihaz ayarı ve Model Yükleme
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

        # Arayüz Bileşenleri
        self.title_label = tk.Label(
            root, text="Sanat Analiz Paneli", 
            font=("Arial", 16, "bold"), fg="#ffffff", bg="#1e1e1e"
        )
        self.title_label.pack(pady=15)

        # Görsel Önizleme Kutusu
        self.image_label = tk.Label(
            root, text="Henüz Görsel Seçilmedi", 
            fg="#888888", bg="#2d2d2d", width=40, height=12, relief="groove"
        )
        self.image_label.pack(pady=10)

        # Buton
        self.select_btn = tk.Button(
            root, text="Görsel Seç ve Analiz Et", command=self.select_image,
            font=("Arial", 11, "bold"), bg="#007acc", fg="white", 
            padx=15, pady=8, relief="flat", cursor="hand2"
        )
        self.select_btn.pack(pady=15)

        # Sonuç Kartı
        self.result_frame = tk.Frame(root, bg="#2d2d2d", padx=15, pady=10)
        self.result_frame.pack(fill="x", padx=20, pady=10)

        self.res_title = tk.Label(
            self.result_frame, text="Sonuç: -", 
            font=("Arial", 12, "bold"), fg="#ffffff", bg="#2d2d2d", anchor="w"
        )
        self.res_title.pack(fill="x")

        self.res_conf = tk.Label(
            self.result_frame, text="Güven Orani: -", 
            font=("Arial", 10), fg="#aaaaaa", bg="#2d2d2d", anchor="w"
        )
        self.res_conf.pack(fill="x", pady=(5, 0))

    def load_model(self):
        model_path = "sanat_modeli.pth"
        if not os.path.exists(model_path):
            messagebox.showerror("Hata", f"'{model_path}' dosyasi bulunamadi! Önce eğitimi tamamlayin.")
            return

        self.model = models.efficientnet_b0()
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Görsel Dosyalari", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        # Görseli Arayüzde Göster
        img = Image.open(file_path)
        img_preview = img.copy()
        img_preview.thumbnail((280, 200))
        img_tk = ImageTk.PhotoImage(img_preview)
        
        self.image_label.config(image=img_tk, text="", width=280, height=200)
        self.image_label.image = img_tk

        # Tahmin Et
        self.predict(file_path)

    def predict(self, image_path):
        if self.model is None:
            return

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        class_names = ['Fully_Human (İnsan Çizimi)', 'Pure_AI (Yapay Zekâ)']

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        res_text = class_names[predicted_idx.item()]
        conf_val = confidence.item() * 100

        # Sonuca göre renk ayarı
        color = "#4caf50" if predicted_idx.item() == 0 else "#ff5252"

        self.res_title.config(text=f"Sonuç: {res_text}", fg=color)
        self.res_conf.config(text=f"Güven Orani: %{conf_val:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ArtClassifierGUI(root)
    root.mainloop()