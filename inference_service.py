from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
from typing import List
from torchvision import transforms
import uvicorn
from pathlib import Path
import zipfile
import os
import shutil
from datetime import datetime

# ✅ KEEP THESE
from class_registry import ClassRegistry
from dynamic_model import ModelManager
from dataset_manager import DatasetManager
from trainer import train_new_model

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ❌ REMOVED: ImageCrawler

# =========================
# GLOBALS
# =========================
registry = None
model_manager = None
dataset_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, model_manager, dataset_manager

    registry = ClassRegistry()
    model_manager = ModelManager()
    dataset_manager = DatasetManager()

    # Load latest model if exists
    await model_manager.load_latest_model(registry)

    yield


app = FastAPI(
    title="Dynamic Image Classification API (LOCAL ONLY)",
    version="3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(380),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# CLASSES
# =========================
@app.get("/classes")
async def get_classes():
    dataset_dir = Path("skin-disease-datasaet/train_set")
    classes = []

    if dataset_dir.exists():
        for d in dataset_dir.iterdir():
            if d.is_dir() and not d.name.startswith("v"):
                count = len(list(d.glob("*.*")))
                classes.append({
                    "name": d.name,
                    "image_count": count
                })

    return {"classes": classes, "total": len(classes)}


@app.post("/classes/add")
async def add_class(class_name: str):
    """LOCAL ONLY - no crawling"""
    class_id = await registry.add_class(class_name)

    dataset_dir = Path("skin-disease-datasaet/train_set") / class_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    return {
        "message": f"Class '{class_name}' created locally",
        "class_id": class_id
    }


# =========================
# DATASET UPLOAD (KEEPED)
# =========================
@app.post("/upload-dataset")
async def upload_dataset(
    class_name: str = None,
    files: List[UploadFile] = File(...)
):
    saved = 0
    base_dir = Path("data/datasets")
    base_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        contents = await file.read()

        if file.filename.endswith(".zip"):
            temp_dir = base_dir / "temp_extract"
            temp_dir.mkdir(exist_ok=True)

            zip_path = temp_dir / file.filename
            with open(zip_path, "wb") as f:
                f.write(contents)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            os.remove(zip_path)

            for root, _, file_list in os.walk(temp_dir):
                for file_name in file_list:
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        src = Path(root) / file_name

                        if "_" in file_name:
                            class_folder = file_name.split("_")[0].lower()
                        else:
                            class_folder = Path(root).name.lower()

                        await registry.add_class(class_folder)

                        dest_dir = base_dir / class_folder
                        dest_dir.mkdir(parents=True, exist_ok=True)

                        shutil.copy2(src, dest_dir / file_name)
                        saved += 1

            shutil.rmtree(temp_dir)

        else:
            filename = Path(file.filename).name

            if "_" in filename:
                class_folder = filename.split("_")[0].lower()
            elif class_name:
                class_folder = class_name.lower()
            else:
                class_folder = "unknown"

            await registry.add_class(class_folder)

            dest_dir = base_dir / class_folder
            dest_dir.mkdir(parents=True, exist_ok=True)

            with open(dest_dir / filename, "wb") as f:
                f.write(contents)

            saved += 1

    return {
        "message": f"{saved} images uploaded (local)",
        "status": "success"
    }


# =========================
# TRAIN
# =========================
@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    dataset_dir = Path("skin-disease-datasaet/train_set")

    classes = [
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith("v")
    ]

    if len(classes) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 classes")

    version_name = f"v{len(classes)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    await dataset_manager.create_version(
        version_name=version_name,
        classes=classes,
        source_dirs=[dataset_dir]
    )

    background_tasks.add_task(
        train_new_model,
        dataset_version=version_name,
        registry=registry,
        model_manager=model_manager
    )

    background_tasks.add_task(model_manager.load_latest_model, registry)

    return {
        "message": "Training started (LOCAL DATASET)",
        "dataset_version": version_name
    }


# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if model_manager.active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    dataset_dir = Path("skin-disease-datasaet/train_set")
    current_classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])

    # ✅ FIX: no crash
    if set(current_classes) != set(model_manager.class_names):
        return {
            "warning": "Model outdated. Retrain required.",
            "current_classes": current_classes
        }

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    tensor = transform(image).unsqueeze(0)
    device = next(model_manager.active_model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model_manager.active_model(tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(model_manager.class_names)))

    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            "class": model_manager.class_names[idx.item()],
            "confidence": round(prob.item(), 4)
        })

    return {"predictions": predictions}


@app.get("/models/status")
async def get_model_status():
    if model_manager.active_model:
        return {
            "status": "loaded",
            "class_count": len(model_manager.class_names),
        }
    return {"status": "not_loaded"}


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/reload")
async def reload_model():
    ok = await model_manager.load_latest_model(registry)
    if ok:
        return {"message": "Model reloaded successfully"}
    return {"message": "No model found"}