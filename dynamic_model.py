import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional
import json
from pathlib import Path
from dataclasses import asdict, dataclass
import asyncio

@dataclass
class ModelConfig:
    model_name: str = "efficientnet-b4"
    input_size: int = 380
    num_classes: int = 10
    pretrained: bool = True
    dropout_rate: float = 0.2

class DynamicEfficientNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        if config.model_name == "efficientnet-b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if config.pretrained else None
            self.base_model = models.efficientnet_b4(weights=weights)
            in_features = 1792
        elif config.model_name == "efficientnet-b2":
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if config.pretrained else None
            self.base_model = models.efficientnet_b2(weights=weights)
            in_features = 1408
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")
        
        # Freeze early layers for transfer learning
        for param in list(self.base_model.parameters())[:-50]:
            param.requires_grad = False
        
        # Replace classifier with dynamic layer
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate, inplace=True),
            nn.Linear(in_features, config.num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def adapt_to_new_classes(self, new_num_classes: int):
        """Dynamically adapt the classifier layer to new number of classes"""
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, new_num_classes)
        self.config.num_classes = new_num_classes
    
    def save(self, path: Path, class_names: List[str], metadata: dict = None):
        """Save model with metadata"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': asdict(self.config),
            'class_names': class_names,
            'metadata': metadata or {}
        }
        torch.save(save_dict, path)
        
        # Also save metadata separately for easy loading
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'class_names': class_names,
                'config': asdict(self.config),
                'metadata': metadata or {}
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: Path, device: torch.device = None):
        """Load model with metadata"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        raw_config = checkpoint.get("config", {})

        model_config = ModelConfig(
            model_name=raw_config.get("model_name", "efficientnet-b4"),
            input_size=raw_config.get("input_size", 380),
            num_classes=len(checkpoint["class_names"]),
            pretrained=raw_config.get("pretrained", True),
            dropout_rate=raw_config.get("dropout_rate", 0.2),
        )

        model = cls(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, checkpoint['class_names']

class ModelManager:
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_model = None
        self.class_names = []
    
    async def load_latest_model(self, registry):
        """Load the latest trained model"""
        model_info = await registry.get_latest_model()
        if model_info:
            model_path = Path(model_info[6])  # path column
            if model_path.exists():
                self.active_model, self.class_names = DynamicEfficientNet.load(model_path)
                return True
        return False
    
    async def reload_if_new(self, registry):
        """Check for and load new model versions"""
        model_info = await registry.get_latest_model()
        if model_info:
            current_hash = await registry.get_class_hash()
            if model_info[2] != current_hash:  # class_list_hash
                return await self.load_latest_model(registry)
        return False