import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import List, Dict
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from dynamic_model import DynamicEfficientNet, ModelConfig

# =========================
# DATASET (MULTI SOURCE)
# =========================
class ImageDataset(Dataset):
    def __init__(self, data_sources: List[Path], class_names, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for label_idx, class_name in enumerate(class_names):
            for src in data_sources:
                class_dir = src / class_name
                if class_dir.exists():
                    for img in class_dir.glob("*"):
                        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            self.images.append(img)
                            self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================
# TRAINER
# =========================
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, class_names, config):
        self.model = model
        self.class_names = class_names
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.scaler = GradScaler() if self.device.type == "cuda" else None

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train(self, save_path):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0

        for epoch in range(10):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for images, labels in tqdm(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

            train_acc = 100 * correct / total

            # VALIDATION
            self.model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)

                    loss = criterion(outputs, labels)

                    v_loss += loss.item()
                    _, pred = outputs.max(1)

                    v_total += labels.size(0)
                    v_correct += pred.eq(labels).sum().item()

                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = 100 * v_correct / v_total

            # SAVE HISTORY
            self.history["train_loss"].append(total_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}, Val Acc {val_acc:.2f}")

            if val_acc > best_acc:
                best_acc = val_acc

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "class_names": self.class_names,
                    "config": {
                        "model_name": "efficientnet-b4",
                        "input_size": 380,
                        "num_classes": len(self.class_names),
                        "pretrained": True,
                        "dropout_rate": 0.2
                    }
                }, save_path)

        # =========================
        # SAVE METRICS + REPORT
        # =========================
        report = classification_report(all_labels, all_preds, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds).tolist()

        metrics = {
            "history": self.history,
            "confusion_matrix": cm,
            "classification_report": report
        }

        with open("data/models/report.json", "w") as f:
            json.dump(metrics, f, indent=2)


# =========================
# MAIN TRAIN FUNCTION
# =========================
async def train_new_model(dataset_version, registry, model_manager, config=None):
    data_sources = [
        Path("skin-disease-datasaet/train_set")
    ]

    class_names = set()
    for src in data_sources:
        if src.exists():
            for d in src.iterdir():
                if d.is_dir():
                    class_names.add(d.name)

    class_names = sorted(list(class_names))

    if len(class_names) < 2:
        raise ValueError("Need at least 2 classes")

    transform = transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(380),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(data_sources, class_names, transform)
    val_dataset = ImageDataset(data_sources, class_names, transform)

    model = DynamicEfficientNet(
        ModelConfig(num_classes=len(class_names))
    )

    trainer = Trainer(model, train_dataset, val_dataset, class_names, {})

    save_path = Path("data/models/model.pth")
    trainer.train(save_path)
    # ✅ REGISTER MODEL
    class_hash = await registry.get_class_hash()

    await registry.register_model_version(
        model_name="efficientnet-dynamic",
        version="v1",
        class_hash=class_hash,
        model_path=str(save_path),
        accuracy=None
    )
    return save_path