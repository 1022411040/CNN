import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil
from dataclasses import dataclass, asdict
import aiofiles
import asyncio

@dataclass
class DatasetVersion:
    version: str
    created_at: datetime
    class_count: int
    total_images: int
    class_distribution: Dict[str, int]
    path: Path

class DatasetManager:
    def __init__(self, base_dir: str = "data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_dir / "versions.json"
        self._load_versions()
    
    def _load_versions(self):
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_versions(self):
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    async def create_version(
        self,
        version_name: str,
        classes: List[str],
        source_dirs: List[Path]
    ) -> DatasetVersion:
        """Create a new dataset version"""
        version_dir = self.base_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        # Create directory structure
        class_distribution = {}
        total_images = 0
        
        for class_name in classes:
            class_dir = version_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copy images from source directories
            image_count = 0
            for source_dir in source_dirs:
                source_class_dir = source_dir / class_name
                if source_class_dir.exists():
                    for img_file in source_class_dir.glob("*.jpg"):
                        dest_file = class_dir / img_file.name
                        shutil.copy2(img_file, dest_file)
                        image_count += 1
            
            class_distribution[class_name] = image_count
            total_images += image_count
        
        # Create version metadata
        version = DatasetVersion(
            version=version_name,
            created_at=datetime.now(),
            class_count=len(classes),
            total_images=total_images,
            class_distribution=class_distribution,
            path=version_dir
        )
        
        # Save metadata
        self.versions[version_name] = asdict(version)
        self.versions[version_name]['path'] = str(version_dir)
        self._save_versions()
        
        # Create train/val/test split file
        await self._create_split_file(version_dir, class_distribution)
        
        return version
    
    async def _create_split_file(self, version_dir: Path, class_distribution: Dict[str, int]):
        """Create train/val/test split configuration"""
        split_config = {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "classes": {}
        }
        
        for class_name, count in class_distribution.items():
            train_count = int(count * 0.7)
            val_count = int(count * 0.15)
            test_count = count - train_count - val_count
            
            split_config["classes"][class_name] = {
                "total": count,
                "train": train_count,
                "val": val_count,
                "test": test_count
            }
        
        split_file = version_dir / "split_config.json"
        async with aiofiles.open(split_file, 'w') as f:
            await f.write(json.dumps(split_config, indent=2))
    
    def get_version(self, version_name: str) -> Optional[DatasetVersion]:
        if version_name in self.versions:
            data = self.versions[version_name]
            return DatasetVersion(
                version=data['version'],
                created_at=datetime.fromisoformat(data['created_at']),
                class_count=data['class_count'],
                total_images=data['total_images'],
                class_distribution=data['class_distribution'],
                path=Path(data['path'])
            )
        return None
    
    def list_versions(self) -> List[str]:
        return list(self.versions.keys())