# routers/recordings.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form, status, BackgroundTasks
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
import uuid
from bson import ObjectId
import tempfile
import os
import httpx
import json
from typing import Optional, List
import logging
import subprocess

router = APIRouter()
logger = logging.getLogger(__name__)

# AI helper functions
async def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file to a temporary WAV file."""
    audio_path = video_path.replace('.webm', '.wav')
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path
        ], check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e}")
        raise Exception(f"Failed to extract audio: {e}")

async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio to text using OpenAI's Whisper API."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            return ""
        
        with open(audio_path, "rb") as audio_file:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": ("audio.wav", audio_file, "audio/wav")},
                    data={"model": "whisper-1"}
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.text}")
                    return ""
                
                return response.json().get("text", "")
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""

async def analyze_transcript(transcript: str) -> List[dict]:
    """Analyze transcript to extract steps with titles and descriptions."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            return []
        
        prompt = f"""
        Extract clearly defined onboarding steps from this transcript of a screen recording.
        For each step, identify a concise title and a description explaining what to do.
        
        Transcript:
        {transcript}
        
        Format the output as a JSON array, with each step having a 'title' and 'description' field.
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are an assistant that extracts structured steps from transcripts."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                }
            )
            
            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.text}")
                return []
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract JSON from the content
            try:
                # Find JSON array in the response
                content = content.strip()
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()
                
                steps = json.loads(content)
                return steps
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response: {content}")
                return []
    except Exception as e:
        logger.error(f"Error analyzing transcript: {e}")
        return []

async def process_recording(file_path: str, file_id: str, db):
    """Process recording to extract steps and update the database."""
    try:
        # Extract audio
        audio_path = await extract_audio_from_video(file_path)
        
        # Transcribe audio
        transcript = await transcribe_audio(audio_path)
        
        # Analyze transcript to get steps
        steps = await analyze_transcript(transcript)
        
        # Update metadata in GridFS
        await db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {
                "metadata.transcript": transcript,
                "metadata.extracted_steps": steps
            }}
        )
        
        # Clean up temporary files
        try:
            os.remove(file_path)
            os.remove(audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")
            
    except Exception as e:
        logger.error(f"Error processing recording: {e}")
        # Update metadata to indicate processing failed
        await db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {"metadata.processing_error": str(e)}}
        )

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
    
    db = request.app.state.db
    fs = AsyncIOMotorGridFSBucket(db)
    
    # Generate a unique filename
    filename = f"{uuid.uuid4()}-step{step_index}.webm"
    
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
    file_id = await fs.upload_from_stream(
        filename,
        open(temp_file_path, "rb").read(),
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

@router.get("/{file_id}")
async def get_recording(request: Request, file_id: str):
    """Retrieve a recording file from GridFS by its ID."""
    from fastapi.responses import StreamingResponse, JSONResponse
    
    db = request.app.state.db
    fs = AsyncIOMotorGridFSBucket(db)
    
    try:
        # Get file info
        file_info = await db.fs.files.find_one({"_id": ObjectId(file_id)})
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get content type from metadata
        metadata = file_info.get("metadata", {})
        content_type = metadata.get("content_type", "video/webm")
        
        # Create a grid_out object to stream the file
        grid_out = await fs.open_download_stream(ObjectId(file_id))
        
        # Return the file as a streaming response
        return StreamingResponse(
            grid_out,
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename={file_info.get('filename')}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")

@router.get("/{file_id}/metadata")
async def get_recording_metadata(request: Request, file_id: str):
    """Get metadata and extracted steps for a recording."""
    db = request.app.state.db
    
    try:
        file_info = await db.fs.files.find_one({"_id": ObjectId(file_id)})
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = file_info.get("metadata", {})
        
        return {
            "file_id": file_id,
            "processing_status": metadata.get("processing_status", "unknown"),
            "step_index": metadata.get("step_index"),
            "extracted_steps": metadata.get("extracted_steps", []),
            "transcript": metadata.get("transcript", ""),
            "error": metadata.get("processing_error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metadata: {str(e)}")

@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(request: Request, file_id: str):
    """Delete a recording file from GridFS."""
    db = request.app.state.db
    fs = AsyncIOMotorGridFSBucket(db)
    
    try:
        await fs.delete(ObjectId(file_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")