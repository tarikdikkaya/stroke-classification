import timm
import torch
import os
import numpy as np
from pathlib import Path
import pydicom
from PIL import Image
from torchvision import transforms
from typing import Dict
from medicai.models.vit import ViTB16  # Import the ViTB16 model class
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm

# Change to the working directory
working_directory = "/content/drive/MyDrive/Teknofest 2025 SYZ Cerebral Complex/codes/stroke-classification"
os.chdir(working_directory)

checkpoint_path = "/content/drive/MyDrive/Teknofest 2025 SYZ Cerebral Complex/models/best_model-epoch=11-val_loss=0.0082.ckpt"

# Load the model using PyTorch Lightning's load_from_checkpoint
model = ViTB16.load_from_checkpoint(
    checkpoint_path,
    num_classes=2,
    strict=False  # Use strict=False if some keys in the checkpoint don't match the model
)
# Set model to evaluation mode
model.eval()

def infer_and_get_results(
    file_path: str,
    model: torch.nn.Module,
    class_names: Dict[int, str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Dosya uzantısına göre DICOM veya JPG/PNG dosyasını işleyip model ile tahmin yapar.
    
    Args:
        file_path: DICOM ya da görüntü dosyasının yolu
        model: Inference için PyTorch modeli
        class_names: Sınıf indekslerini isimlerine eşleyen sözlük
        device: Çalıştırılacak cihaz ('cuda' veya 'cpu')
        
    Returns:
        Tahmin sonuçlarını içeren sözlük
    """
    # ViT model 384x384 giriş boyutu kullanıyor
    image_size = 384
    
    # Modelin beklediği girdi için dönüşüm tanımlaması
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dosya uzantısına göre dosyayı aç
    ext = Path(file_path).suffix.lower()
    if ext == '.dcm':
        # DICOM dosyası ise pydicom ile oku
        dicom = pydicom.dcmread(file_path, force=True)
        image = dicom.pixel_array.astype(np.float32)
    else:
        # JPG, JPEG, PNG gibi dosyalar için PIL kullanarak oku
        img = Image.open(file_path)
        # Eğer görüntü alfa kanallı ise RGB'ye dönüştür
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # Gerekirse grayscale'den 3 kanallı hale getir
        if img.mode == 'L':
            img = img.convert('RGB')
        image = np.array(img).astype(np.float32)
    
    # Eğer görüntü tek kanallı ise (2 boyutlu) 3 kanala dönüştür
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize et: [0, 255] aralığına ölçekle
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    
    # PIL görüntüsüne çevir ve dönüşümü uygula
    image = transforms.ToPILImage()(image)
    image_tensor = transform(image)
    
    # Batch boyutunu ekle
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    
    # Uygun cihaza taşı
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    
    # Modeli değerlendirme moduna al
    model.eval()
    
    # Inference işlemi
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Tahmini sınıfı al
        _, predicted_class = torch.max(probabilities, 1)
        confidence = probabilities[0][predicted_class].item()
        predicted_idx = predicted_class.item()
    
    # Sonuç sözlüğünü hazırla
    results = {
        "predicted_class": predicted_idx,
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy(),
        "file": file_path
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Sınıf isimleri tanımı
    class_names = {0: "Normal", 1: "Abnormal"}
    
    # Test veri seti yolu
    test_dir = "/content/drive/MyDrive/Teknofest 2025 SYZ Cerebral Complex/dataset/test"
    
    # Tahmin ve gerçek değerleri depolamak için listeler
    all_preds = []
    all_true = []
    all_results = []
    
    # Test dizini ve alt dizinlerinden tüm dosyaları al (.dcm, .jpg, .jpeg, .png)
    image_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files in {test_dir}")
    
    # Her dosyayı işlemek için tqdm ile ilerleme çubuğu kullan
    for file_path in tqdm(image_files, desc="Processing image files"):
        # Gerçek sınıfı dizin yapısından belirle
        # Örneğin: split_dataset_latest/test/class_0/ dizin yapısına göre
        parts = file_path.split(os.path.sep)
        try:
            test_index = parts.index('test')
        except ValueError:
            continue
        if test_index + 1 < len(parts):
            try:
                # Klasör adı 'class_0' veya 'class_1' ise gerçek sınıfı belirle
                class_folder = parts[test_index + 1]
                if class_folder.startswith('class_'):
                    true_class = int(class_folder.split('_')[1])
                    # Inference yap
                    result = infer_and_get_results(file_path, model, class_names)
                    
                    # Sonuçları depola
                    pred_class = result["predicted_class"]
                    all_preds.append(pred_class)
                    all_true.append(true_class)
                    all_results.append(result)
            except (ValueError, IndexError):
                continue
    
    # Metriği hesapla
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average='weighted')
    recall = recall_score(all_true, all_preds, average='weighted')
    f1 = f1_score(all_true, all_preds, average='weighted')
    
    # Sonuçları yazdır
    print("\n===== Validation Statistics =====")
    print(f"Total samples: {len(all_preds)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix oluştur
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_names[i] for i in range(len(class_names))],
                yticklabels=[class_names[i] for i in range(len(class_names))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Confusion Matrix'i kaydet
    output_dir = "output/validation_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

