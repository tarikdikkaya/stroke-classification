import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
import numpy as np
import pydicom
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 📌 Veri seti yolları
train_path = "split_dataset_latest/train"
test_path = "split_dataset_latest/test"

# 📌 Kullanıcıdan epoch sınırı alma
stop_epoch = int(input("Kaçıncı epoch'ta eğitimi durdurmak istiyorsunuz? "))

# 📌 Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- #
# 📌 DICOM Veri Seti Tanımlama
# ------------------------- #
class DICOMDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Sınıfları otomatik keşfet
        class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_map = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}

        for class_name, label in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for file_name in os.listdir(class_path):
                    dicom_path = os.path.join(class_path, file_name)
                    if os.path.isfile(dicom_path) and file_name.endswith(".dcm"):
                        try:
                            dicom = pydicom.dcmread(dicom_path, force=True)
                            if hasattr(dicom, "pixel_array"):  # DICOM görüntü içeriyor mu?
                                self.samples.append((dicom_path, label))
                        except Exception as e:
                            print(f"⚠️ Hata: {dicom_path} dosyası okunamadı! Hata: {e}")
        
        if len(self.samples) == 0:
            raise ValueError(f"❌ Hata: '{root_dir}' içinde hiç kullanılabilir DICOM dosyası yok!")

        print(f"✅ {len(self.samples)} adet DICOM dosyası yüklendi. Sınıflar: {class_map}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dicom_path, label = self.samples[idx]
        dicom = pydicom.dcmread(dicom_path, force=True)
        image = dicom.pixel_array.astype(np.float32)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# 📌 Veri dönüşümleri (Data Augmentation içerir)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 📌 Veri yükleme
train_dataset = DICOMDataset(root_dir=train_path, transform=transform)
test_dataset = DICOMDataset(root_dir=test_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

# ------------------------- #
# 📌 EfficientNet-L2 (SAM) Modeli
# ------------------------- #
model = timm.create_model("tf_efficientnet_l2.ns_jft_in1k", pretrained=True, num_classes=2)
model.to(device)

# 📌 Loss, Optimizer ve Scheduler
class_weights = torch.tensor([3.35, 1.43]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = 0.0
best_model_path = "best_model_effnetl2.pth"

def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=100):
    global best_acc, best_model_path
    model.train()
    for epoch in range(epochs):
        print(f"\n🔵 [Epoch {epoch+1}/{epochs}] Başladı...")
        running_loss = 0.0
        correct, total = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=f"{running_loss / len(train_loader):.4f}")
        
        accuracy = 100 * correct / total
        scheduler.step()
        print(f"🔹 [Epoch {epoch+1} Tamamlandı] | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")
        
        # 📌 Checkpoint Yönetimi (En iyi modeli sakla)
        if epoch + 1 >= 10:  # 10. epoch'tan sonra checkpoint al
            if accuracy > best_acc:
                best_acc = accuracy
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)  # Önceki checkpoint sil
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Yeni en iyi model kaydedildi (Accuracy: {best_acc:.2f}%)")
        
        # 📌 Kullanıcının belirlediği epoch'ta durdur
        if epoch + 1 == stop_epoch:
            print(f"⏹️ {stop_epoch}. epoch'ta eğitim durduruldu!")
            break

train_model(model, train_loader, criterion, optimizer, scheduler, stop_epoch)
print(f"✅ Eğitim tamamlandı! En iyi model '{best_model_path}' olarak kaydedildi.")
