import sys
import io
import os
import warnings
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import random
from alm import AudioLanguageModel

def convert_numpy_types(obj):
    """Convert numpy and PyTorch types to native Python types for JSON serialization"""
    # Handle PyTorch types first
    try:
        import torch
        if isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass

    # Handle NumPy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# Initialize FastAPI app
app = FastAPI(title="Audio Language Model API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once on startup)
_alm_instance: Optional[AudioLanguageModel] = None

# Store processed audio results by session ID
_audio_results_store: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    model_used: Optional[str] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global _alm_instance
    try:
        print("🚀 Loading AudioLanguageModel...")
        _alm_instance = AudioLanguageModel(
            transcription_device="cuda",
            diarization_device="cuda",
            load_chat=True,
        )
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Audio Language Model API is running",
        "model_loaded": _alm_instance is not None,
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    cuda_health = False

    return {
        "status": "healthy",
        "model_loaded": _alm_instance is not None,
        "chat_available": _alm_instance.chat_model is not None if _alm_instance else False,
        "cuda_healthy": cuda_health,
    }


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """Process an audio file from the frontend.

    Returns:
        - session_id: Unique ID to reference this processed audio
        - results: Complete audio processing results
        - cached: Whether the results came from cache
    """
    if _alm_instance is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")

    # Validate file type
    content_type = file.content_type
    if content_type and not content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Read file content once
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Compute deterministic hash of the audio content
    file_hash = hashlib.sha256(content).hexdigest()
    hash_key = f"audio_hash:{file_hash}"

    # No usable cache → process as new audio
    session_id = str(uuid.uuid4())

    # Save uploaded file temporarily (for the model API)
    temp_dir = Path(tempfile.gettempdir())
    temp_file_path = temp_dir / f"audio_{session_id}_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)

        print(f"📁 Processing audio file: {file.filename} (session: {session_id})")
        print("🔄 Starting audio processing.")

        results = _alm_instance.process_audio(
            str(temp_file_path),
            save_preprocessed_wav=False,
            min_speakers=1,
            max_speakers=10,
        )
        print("✅ Audio processing completed")

        # Store results with session ID in memory
        _audio_results_store[session_id] = results
        print(f"💾 Stored results in memory for session: {session_id}")

        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()

        serializable_results = convert_numpy_types(results)
        return {
            "session_id": session_id,
            "filename": file.filename,
            "results": serializable_results,
            "cached": False,
        }

    except Exception as e:
        if temp_file_path.exists():
            temp_file_path.unlink()

        error_msg = f"Error processing audio: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat about processed audio using the session ID."""
    if _alm_instance is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")

    if _alm_instance.chat_model is None:
        raise HTTPException(status_code=503, detail="Chat model not available. Please check GEMINI_API_KEY.")
    request.session_id=12131
    print(
        f"\n📝 Chat request received - Session: {request.session_id}, "
        f"Question: {request.question}"
    )
    audio_results = None
    if audio_results:
        pass
    else:
        if request.session_id in _audio_results_store:
            audio_results = _audio_results_store[request.session_id]
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Session ID not found: {request.session_id}. "
                    "Please process audio first."
                ),
            )

    try:
        response = _alm_instance.chat(
            question=request.question,
            audio_results=audio_results,
        )

        #await set_json(chat_key, response, ttl=3600)
        if isinstance(response, dict) and response.get("error"):
            raise HTTPException(status_code=500, detail=response["error"])

        return ChatResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error during chat: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a processed audio session
    """

    # Delete in-memory store
    if session_id in _audio_results_store:
        del _audio_results_store[session_id]
        return {"message": f"Session {session_id} deleted successfully"}

    raise HTTPException(status_code=404, detail=f"Session ID not found: {session_id}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
