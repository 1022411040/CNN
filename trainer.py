import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, List, Optional
import json
from PIL import Image
import asyncio
from tqdm import tqdm
import numpy as np
from fastapi import HTTPException
from dataset_manager import DatasetManager
from dynamic_model import DynamicEfficientNet, ModelConfig
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class ImageDataset(Dataset):
    def __init__(self, data_dir: Path, class_names: List[str], transform=None, split: str = 'train'):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.split = split
        
        # Build image paths and labels
        self.images = []
        self.labels = []
        
        for label_idx, class_name in enumerate(class_names):
            class_dir = data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob("*.jpg"):
                    self.images.append(img_file)
                    self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dir: Path,
        val_dir: Path,
        class_names: List[str],
        config: Dict
    ):
        self.model = model
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.class_names = class_names
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.get('mixed_precision', True) and self.device.type == 'cuda' else None
        
        # Setup distributed training if multiple GPUs
        self.world_size = 1
        if self.world_size > 1:
            self.setup_distributed()
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.get('input_size', 380)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(config.get('input_size', 380) + 32),
            transforms.CenterCrop(config.get('input_size', 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def create_data_loaders(self):
        train_dataset = ImageDataset(
            self.train_dir, self.class_names, self.train_transform, 'train'
        )
        val_dataset = ImageDataset(
            self.val_dir, self.class_names, self.val_transform, 'val'
        )
        
        batch_size = self.config.get('batch_size', 32)
        num_workers = 0
        
        if self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=self.world_size, shuffle=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=train_sampler, num_workers=num_workers,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=num_workers,
                pin_memory=True
            )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
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
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, save_path: Path):
        train_loader, val_loader = self.create_data_loaders()
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=1e-4
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.get('epochs', 20)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        self.best_acc = 0
        
        for epoch in range(self.config.get('epochs', 20)):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, epoch + 1
            )
            
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            print(f"Epoch {epoch + 1}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            scheduler.step()
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(save_path, val_acc)

    
    def save_checkpoint(self, path: Path, accuracy: float):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            'accuracy': accuracy,
            'class_names': self.class_names,
            'config': self.config
        }
        torch.save(checkpoint, path)

async def train_new_model(
    dataset_version: str,
    registry,
    model_manager,
    config: Dict = None
):
    """Orchestrate model training pipeline"""
    if config is None:
        config = {
            'epochs': 2,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'mixed_precision': True,
            'input_size': 380
        }
    
    # Get current classes
    class_names = await registry.get_class_names()
    
    # Get dataset path
    dataset_manager = DatasetManager()
    version_info = dataset_manager.get_version(dataset_version)
    
    if not version_info:
        raise ValueError(f"Dataset version {dataset_version} not found")
    
    # ===== DATASET SANITY CHECK =====
    total_images = 0
    for class_name in class_names:
        class_dir = version_info.path / class_name
        if class_dir.exists():
            total_images += len(list(class_dir.glob("*.jpg")))

    if total_images == 0:
        print("Training skipped: No images found. Wait for image crawling to finish.")
        return None


    MIN_IMAGES_PER_CLASS = 5
    for class_name in class_names:
        class_dir = version_info.path / class_name
        count = len(list(class_dir.glob("*.jpg")))
        if count < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class '{class_name}' has only {count} images. "
                f"Minimum required: {MIN_IMAGES_PER_CLASS}"
            )
    # =================================

    # Initialize model
    model_config = ModelConfig(
        model_name="efficientnet-b4",
        input_size=config['input_size'],
        num_classes=len(class_names)
    )
    model = DynamicEfficientNet(model_config)
    
    # Train
    trainer = Trainer(
        model=model,
        train_dir=version_info.path,
        val_dir=version_info.path,  # In production, use separate val set
        class_names=class_names,
        config=config
    )
    
    # Save model
    model_dir = Path("data/models") / f"model_v{len(class_names)}_{dataset_version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    
    trainer.train(model_path)
    
    # Register model
    class_hash = await registry.get_class_hash()
    await registry.register_model_version(
        model_name="efficientnet-dynamic",
        version=f"v{len(class_names)}",
        class_hash=class_hash,
        model_path=str(model_path),
        accuracy=trainer.best_acc
    )
    
    # Reload in model manager
    await model_manager.load_latest_model(registry)
    
    return model_path