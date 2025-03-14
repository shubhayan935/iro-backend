# routers/recordings.py
import os
import logging
import json
import re
import tempfile
from typing import List, Optional, Dict, Any
from bson import ObjectId

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form, status, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

import models_loader
from routers.video_analyzer import OnboardingVideoAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)

# Add an endpoints to check/initiate model loading
@router.get("/status")
async def model_status():
    """Check the status of ML models loading"""
    return models_loader.get_model_status()

@router.post("/load-models")
async def load_models():
    """Manually trigger model loading"""
    models = await models_loader.load_models_async()
    return {
        "status": "Models loading initiated/refreshed",
        "model_status": models_loader.get_model_status()
    }

# Process a recording and extract steps
async def process_recording(file_path: str, file_id: str, db):
    """Process recording to extract onboarding steps and update the database."""
    try:
        logger.info(f"Processing recording {file_id} at {file_path}")

        # Check if we have the necessary models
        if models_loader.get_model("transcription_model") is None:
            raise ValueError("Transcription model not available")
            
        # Extract steps from the video
        logger.info("Starting extraction process")
        steps = await extract_steps_from_recording(file_path)
        logger.info(f"Extracted {len(steps) if isinstance(steps, list) else 'unknown'} steps")

        # Update metadata in GridFS
        await db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {
                "metadata.extracted_steps": steps,
                "metadata.processing_status": "complete"
            }}
        )
        
        # Clean up the temporary video file
        try:
            os.remove(file_path)
            logger.info(f"Deleted temporary file {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")
            
    except Exception as e:
        logger.error(f"Error processing recording: {str(e)}")
        # Update metadata to indicate processing failed
        await db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {"metadata.processing_error": str(e), "metadata.processing_status": "failed"}}
        )
        raise

# Extract steps from a video file
async def extract_steps_from_recording(video_path: str):
    """
    Asynchronously extract steps from a recording.
    Uses threadpool to avoid blocking the event loop.
    """
    try:
        logger.info(f"Creating analyzer for {video_path}")
        analyzer = OnboardingVideoAnalyzer(video_path)
        logger.info("Starting analysis process")
        result = await run_in_threadpool(analyzer.process)
        logger.info("Analysis complete")
        return result
    except Exception as e:
        logger.error(f"Error in extraction: {str(e)}")
        raise

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_recording(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    step_index: str = Form(...)
):
    """Upload a recording file to MongoDB GridFS and process it with AI."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only video files are accepted."
        )
    
    try:
        db = request.app.state.db
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket
        fs = AsyncIOMotorGridFSBucket(db)
        
        # Generate a unique filename
        filename = f"{os.urandom(8).hex()}-step{step_index}.webm"
        
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Store metadata
        metadata = {
            "content_type": file.content_type,
            "step_index": int(step_index),
            "processing_status": "pending"
        }
        
        # Upload to GridFS
        with open(temp_file_path, "rb") as f_in:
            file_bytes = f_in.read()
        file_id = await fs.upload_from_stream(
            filename,
            file_bytes,
            metadata=metadata
        )
        
        # Schedule background processing
        background_tasks.add_task(
            process_recording,
            temp_file_path,
            str(file_id),
            db
        )
        
        # Create the URL for accessing the file
        file_url = f"/recordings/{file_id}"
        
        return {
            "url": file_url,
            "file_id": str(file_id),
            "processing_status": "pending"
        }
    except Exception as e:
        logger.error(f"Error uploading recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{file_id}")
async def get_recording(request: Request, file_id: str):
    """Retrieve a recording file from GridFS by its ID."""
    try:
        db = request.app.state.db
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket
        fs = AsyncIOMotorGridFSBucket(db)
        
        file_info = await db.fs.files.find_one({"_id": ObjectId(file_id)})
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = file_info.get("metadata", {})
        content_type = metadata.get("content_type", "video/webm")
        
        grid_out = await fs.open_download_stream(ObjectId(file_id))
        
        return StreamingResponse(
            grid_out,
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename={file_info.get('filename')}"}
        )
    except Exception as e:
        logger.error(f"Error retrieving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")

@router.get("/{file_id}/metadata")
async def get_recording_metadata(request: Request, file_id: str):
    """Get metadata and extracted steps for a recording."""
    try:
        db = request.app.state.db
        file_info = await db.fs.files.find_one({"_id": ObjectId(file_id)})
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = file_info.get("metadata", {})
        return {
            "file_id": file_id,
            "processing_status": metadata.get("processing_status", "unknown"),
            "step_index": metadata.get("step_index"),
            "extracted_steps": metadata.get("extracted_steps", []),
            "error": metadata.get("processing_error")
        }
    except Exception as e:
        logger.error(f"Error retrieving metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata: {str(e)}")

@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(request: Request, file_id: str):
    """Delete a recording file from GridFS."""
    try:
        db = request.app.state.db
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket
        fs = AsyncIOMotorGridFSBucket(db)
        await fs.delete(ObjectId(file_id))
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")