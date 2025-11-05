#!/usr/bin/env python3
"""
Pre-download Whisper models during Docker build to eliminate cold start delays.
This script downloads the model specified by environment variables or defaults.
"""
import os
import sys
from pathlib import Path

# Add the app directory to Python path to import our modules
sys.path.insert(0, '/app')

try:
    from faster_whisper import WhisperModel
    from app.config import CONFIG
    print(f"Starting model download...")
    print(f"Model: {CONFIG.MODEL_NAME}")
    print(f"Device: {CONFIG.DEVICE}")
    print(f"Quantization: {CONFIG.MODEL_QUANTIZATION}")
    print(f"Download path: {CONFIG.MODEL_PATH}")
    
    # Ensure the model directory exists
    os.makedirs(CONFIG.MODEL_PATH, exist_ok=True)
    
    # Download the model (this will cache it to MODEL_PATH)
    print("Downloading model... This may take a few minutes.")

    # Prepare model kwargs with HF token if available
    model_kwargs = {
        "model_size_or_path": CONFIG.MODEL_NAME,
        "device": "cpu",  # Use CPU during build to avoid GPU requirements
        "compute_type": "int8",  # Use int8 during build for efficiency
        "download_root": CONFIG.MODEL_PATH,
        "cpu_threads": 4,  # Optimize for container build
        "num_workers": 1   # Single worker during build
    }

    # Add HF token if provided (for private/gated models)
    if CONFIG.HF_TOKEN:
        print("Using Hugging Face authentication token")
        model_kwargs["token"] = CONFIG.HF_TOKEN

    model = WhisperModel(**model_kwargs)
    
    print(f"✅ Model '{CONFIG.MODEL_NAME}' successfully downloaded to {CONFIG.MODEL_PATH}")
    print(f"Model files:")
    
    # List the downloaded files
    model_path = Path(CONFIG.MODEL_PATH)
    if model_path.exists():
        for file in model_path.rglob("*"):
            if file.is_file():
                print(f"  - {file.relative_to(model_path)} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Test that the model can be loaded
    print("Testing model loading...")
    del model  # Free memory
    print("✅ Model pre-download completed successfully!")
    
except Exception as e:
    print(f"❌ Error downloading model: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)