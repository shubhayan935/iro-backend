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
import shutil

router = APIRouter()
logger = logging.getLogger(__name__)

# NEW HELPER FUNCTION: extract key frames from video using ffmpeg
def extract_key_frames(video_path: str, output_dir: str, fps: float = 1/10) -> List[str]:
    """
    Extract key frames from the video at the specified frames per second (fps)
    and save them in output_dir. Returns a list of file paths for the extracted frames.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Example command: extract one frame every 10 seconds
    output_pattern = os.path.join(output_dir, "frame_%03d.jpg")
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            output_pattern
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting key frames: {e}")
        raise Exception(f"Failed to extract key frames: {e}")
    
    # List all extracted frames (sorted by filename)
    frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return frames

# NEW HELPER FUNCTION: dummy captioning function for an image
def caption_image(image_path: str) -> str:
    """
    Dummy image captioning function.
    In a real system, you would integrate with an image captioning model/API here.
    """
    # For now, return a placeholder caption that includes the filename.
    return f"Caption for {os.path.basename(image_path)}"

# NEW FUNCTION: extract onboarding steps directly from video (and audio) context
async def extract_steps_from_recording(video_path: str) -> List[dict]:
    """
    Extract onboarding steps from the video recording by:
      1. Extracting key frames.
      2. Generating captions for those frames.
      3. Compiling the captions into a prompt for an LLM to extract steps.
      
    Each step should include:
        - title: A clear title summarizing the step.
        - description: A detailed explanation of what to do.
        - ui_elements: A list of UI elements (e.g., buttons, fields, menus) to look for.
        - inputs: Any specific inputs or selections required.
        - success_criteria: Criteria for successful completion.
    """
    # Create a temporary directory to store key frames
    temp_dir = tempfile.mkdtemp()
    try:
        # Extract key frames (for example, one frame every 10 seconds)
        frames = extract_key_frames(video_path, temp_dir, fps=1)
        if not frames:
            raise Exception("No key frames extracted from the video.")

        # Generate captions for each frame (in a real system, use an image captioning API)
        captions = []
        for frame in frames:
            caption = caption_image(frame)
            captions.append(caption)

        # Build a prompt that provides the key frame captions to the LLM
        prompt = (
            "You are an assistant that extracts structured onboarding steps from a screen recording. "
            "Below are captions generated from key frames of a video recording where an admin user walked through an onboarding process. "
            "Based on these captions, extract the onboarding steps. For each step, provide:\n"
            "1. A clear title summarizing the step.\n"
            "2. A detailed description explaining exactly what to do.\n"
            "3. A list of UI elements to look for (e.g., buttons, fields, menus).\n"
            "4. Any specific inputs or selections that need to be made.\n"
            "5. Success criteria for completing the step.\n\n"
            "Key frame captions:\n"
        )
        for idx, cap in enumerate(captions, 1):
            prompt += f"{idx}. {cap}\n"
        prompt += "\nFormat your answer as a JSON array of objects with the following keys: "
        prompt += '"title", "description", "ui_elements", "inputs", "success_criteria".'

        # Call OpenAI API to extract steps using the prompt
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found")
            return []

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an assistant that extracts structured steps from video recordings."},
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
            # Try to extract JSON from the response
            try:
                content = content.strip()
                if content.startswith("```json"):
                    content = content.split("```json")[1].split("```")[0].strip()
                elif content.startswith("```"):
                    content = content.split("```")[1].split("```")[0].strip()
                steps = json.loads(content)
                return steps
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {content}")
                return []
    except Exception as e:
        logger.error(f"Error extracting steps from recording: {e}")
        return []
    finally:
        # Clean up temporary key frames directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up key frames directory: {e}")

# Modified process_recording function (now using the new extraction function)
async def process_recording(file_path: str, file_id: str, db):
    """Process recording to extract onboarding steps and update the database."""
    try:
        # Instead of transcribing, directly extract steps from the video
        steps = await extract_steps_from_recording(file_path)
        
        # Update metadata in GridFS with the extracted steps
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
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")
            
    except Exception as e:
        logger.error(f"Error processing recording: {e}")
        # Update metadata to indicate processing failed
        await db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {"metadata.processing_error": str(e), "metadata.processing_status": "failed"}}
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
