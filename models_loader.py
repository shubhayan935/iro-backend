# models_loader.py
import os
import ssl
import logging
from typing import Dict, Any, Optional
import time
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix SSL certificate issues
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context  # This bypasses certificate verification

# Global variables to store models
models = {
    "transcription_model": None,
    "nlp_summarizer": None,
    "sentence_model": None,
    "ui_detector": None,
    "cursor_tracker": None
}

model_status = {
    "loaded": False,
    "loading": False,
    "error": None
}

async def load_models_async():
    """
    Load ML models asynchronously for video processing.
    Uses deferred imports to prevent startup crashes.
    """
    global models, model_status
    
    # If already loading or loaded successfully, don't do it again
    if model_status["loading"]:
        logger.info("Models already loading, returning")
        return models
        
    if model_status["loaded"] and not model_status["error"]:
        logger.info("Models already loaded, returning")
        return models
        
    model_status["loading"] = True
    
    try:
        # Import heavy libraries only when needed
        from fastapi.concurrency import run_in_threadpool
        
        # First load lightweight models that won't crash the app
        logger.info("Loading lightweight components first...")
        
        # UI Element Detector (lightweight)
        from routers.ui_helpers import UIElementDetector, CursorTracker
        models["ui_detector"] = UIElementDetector()
        logger.info("✓ Loaded UI detector")
        
        models["cursor_tracker"] = CursorTracker()
        logger.info("✓ Loaded cursor tracker")
        
        # Load whisper with exception handling and threadpool
        logger.info("Loading whisper model (tiny)...")
        try:
            import whisper
            # Wrap in threadpool to prevent blocking
            models["transcription_model"] = await run_in_threadpool(
                lambda: whisper.load_model("tiny")
            )
            logger.info("✓ Loaded whisper model")
        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            model_status["error"] = f"Whisper error: {str(e)}"
        
        # Optional models - can fail gracefully
        # try:
        #     logger.info("Loading NLP models (may take a moment)...")
        #     from transformers import pipeline
        #     # Use a lightweight model variant
        #     models["nlp_summarizer"] = pipeline(
        #         "summarization", 
        #         model="facebook/bart-large-cnn", 
        #         device=-1  # Use CPU to prevent CUDA issues
        #     )
        #     logger.info("✓ Loaded NLP models")
        # except Exception as e:
        #     logger.warning(f"NLP models not available: {e}")
        
        # try:
        #     logger.info("Loading sentence transformer...")
        #     from sentence_transformers import SentenceTransformer
        #     models["sentence_model"] = SentenceTransformer('all-MiniLM-L6-v2')
        #     logger.info("✓ Loaded sentence transformer")
        # except Exception as e:
        #     logger.warning(f"Sentence transformer not available: {e}")
        
        # Mark as loaded even if some optional models failed
        model_status["loaded"] = True
        model_status["loading"] = False
        logger.info("Model loading complete")
        
    except Exception as e:
        # Catch any unexpected errors during load
        logger.error(f"Unexpected error loading models: {e}")
        model_status["error"] = str(e)
        model_status["loading"] = False
    
    return models

def get_model(name: str) -> Any:
    """
    Safely get a model by name, with better error handling.
    
    Args:
        name: Name of the model to retrieve
        
    Returns:
        The requested model or None if not available
    """
    if name not in models:
        logger.warning(f"Requested unknown model: {name}")
        return None
        
    if models[name] is None:
        logger.warning(f"Model {name} not loaded yet")
        
    return models[name]

def get_model_status() -> Dict[str, Any]:
    """Get the current status of model loading"""
    return {
        "loaded": model_status["loaded"],
        "loading": model_status["loading"],
        "error": model_status["error"],
        "available_models": [name for name, model in models.items() if model is not None]
    }