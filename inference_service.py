from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
from typing import List, Dict
import asyncio
from contextlib import asynccontextmanager
from torchvision import transforms
import uvicorn
from pathlib import Path
import json
from datetime import datetime


# Import our components
from class_registry import ClassRegistry
from dynamic_model import ModelManager, DynamicEfficientNet
from image_crawler import ImageCrawler
from dataset_manager import DatasetManager
from trainer import train_new_model
from fastapi.middleware.cors import CORSMiddleware
from neuron_monitor import NeuronMonitor


# Global components
registry = None
model_manager = None
crawler = None
dataset_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global registry, model_manager, crawler, dataset_manager
    
    registry = ClassRegistry()
    model_manager = ModelManager()
    crawler = ImageCrawler()
    dataset_manager = DatasetManager()
    
    # Load latest model
    await model_manager.load_latest_model(registry)
    
    yield
    
    # Shutdown
    pass

app = FastAPI(
    title="Dynamic Image Classification API",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Transform for inference
transform = transforms.Compose([
    transforms.Resize(380),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/classes")
async def get_classes():
    """Get all available classes"""
    classes = await registry.get_all_classes()
    return {
        "classes": [{"id": c.id, "name": c.name, "image_count": c.image_count} for c in classes],
        "total": len(classes)
    }

@app.post("/classes/add")
async def add_class(class_name: str, background_tasks: BackgroundTasks):
    """Add a new class to the system"""
    class_id = await registry.add_class(class_name)
    
    # Start background crawling for this class
    background_tasks.add_task(
        crawler.crawl_class,
        class_name=class_name,
        max_images=500
    )
    
    return {
        "message": f"Class '{class_name}' added successfully",
        "class_id": class_id,
        "crawling_started": True
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """Make prediction on uploaded image"""
    if model_manager.active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read and process image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transforms
    tensor = transform(image).unsqueeze(0)
    
    # Move to device
    device = next(model_manager.active_model.parameters()).device
    tensor = tensor.to(device)
    
    # Inference
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model_manager.active_model(tensor)
        else:
            outputs = model_manager.active_model(tensor)
        
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(model_manager.class_names)))
    
    # Prepare response
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append({
            "class": model_manager.class_names[idx.item()],
            "confidence": round(prob.item(), 4),
            "class_id": idx.item() + 1
        })
    
    return {
        "predictions": predictions,
        "top_class": predictions[0]["class"],
        "top_confidence": predictions[0]["confidence"],
        "model_version": "dynamic",
        "total_classes": len(model_manager.class_names)
    }

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    classes = await registry.get_class_names()
    if len(classes) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 classes to train"
        )

    # 🔒 BLOCK TRAINING IF CRAWLING STILL RUNNING
    if crawler.active_crawls:
        raise HTTPException(
            status_code=409,
            detail=f"Crawling still running for: {list(crawler.active_crawls)}"
        )

    version_name = f"v{len(classes)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    source_dir = Path("data/datasets")

    await dataset_manager.create_version(
        version_name=version_name,
        classes=classes,
        source_dirs=[source_dir]
    )

    background_tasks.add_task(
        train_new_model,
        dataset_version=version_name,
        registry=registry,
        model_manager=model_manager
    )

    return {
        "message": "Training started",
        "dataset_version": version_name
    }

@app.get("/models/status")
async def get_model_status():
    """Get current model status"""
    if model_manager.active_model:
        return {
            "status": "loaded",
            "class_count": len(model_manager.class_names),
            "device": str(next(model_manager.active_model.parameters()).device),
            "classes": model_manager.class_names
        }
    else:
        return {"status": "not_loaded"}

@app.post("/reload")
async def reload_model():
    """Force reload of latest model"""
    success = await model_manager.load_latest_model(registry)
    if success:
        return {"message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail="No trained model found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.active_model is not None,
        "class_count": len(model_manager.class_names) if model_manager.class_names else 0,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "inference_service:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        reload=False
    )
