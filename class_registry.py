import sqlite3
import json
import aiosqlite
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import hashlib

@dataclass
class ClassRecord:
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    dataset_version: str
    image_count: int = 0
    is_active: bool = True

class ClassRegistry:
    def __init__(self, db_path: str = "data/classes.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dataset_version TEXT DEFAULT 'v1.0',
                    image_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    class_list_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accuracy REAL,
                    path TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(model_name, version)
                )
            """)
    
    async def add_class(self, class_name: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR IGNORE INTO classes (name, dataset_version)
                VALUES (?, 'v1.0')
            """, (class_name,))
            await db.commit()
            
            cursor = await db.execute(
                "SELECT id FROM classes WHERE name = ?",
                (class_name,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None
    
    async def get_all_classes(self) -> List[ClassRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM classes WHERE is_active = TRUE ORDER BY name"
            )
            rows = await cursor.fetchall()
            return [
                ClassRecord(
                    id=row[0],
                    name=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    updated_at=datetime.fromisoformat(row[3]),
                    dataset_version=row[4],
                    image_count=row[5],
                    is_active=bool(row[6])
                ) for row in rows
            ]
    
    async def get_class_names(self) -> List[str]:
        classes = await self.get_all_classes()
        return [c.name for c in classes]
    
    async def get_class_hash(self) -> str:
        classes = await self.get_class_names()
        class_str = ",".join(sorted(classes))
        return hashlib.md5(class_str.encode()).hexdigest()
    
    async def register_model_version(
        self,
        model_name: str,
        version: str,
        class_hash: str,
        model_path: str,
        accuracy: float = None
    ):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO model_versions 
                (model_name, version, class_list_hash, path, accuracy)
                VALUES (?, ?, ?, ?, ?)
            """, (model_name, version, class_hash, model_path, accuracy))
            await db.commit()
    
    async def get_latest_model(self, model_name: str = "efficientnet-dynamic"):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM model_versions 
                WHERE model_name = ? AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """, (model_name,))
            row = await cursor.fetchone()
            return row