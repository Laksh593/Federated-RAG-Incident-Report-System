from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from rag.incident_report import IncidentReportGenerator
from rag.config import config
import asyncio
import sys
import os

# Import the new inference utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'model')))
try:
    from inference import load_cfl_model, predict_anomaly
except ImportError:
    print("Warning: Could not import inference model.")

# Global instance
rag_system = None
pytorch_model = None
pytorch_config = None
pytorch_device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system, pytorch_model, pytorch_config, pytorch_device
    print("Initializing RAG System...")
    rag_system = IncidentReportGenerator()
    rag_system.initialize_knowledge_base()
    
    # Load PyTorch Model for Action Recognition
    print("Initializing PyTorch Anomaly Detection Model...")
    try:
        pytorch_model, pytorch_config, pytorch_device = load_cfl_model()
    except Exception as e:
        print(f"❌ Failed to load PyTorch model on startup: {e}")
        raise RuntimeError(f"PyTorch anomaly model could not be initialized: {e}")
        
    yield
    print("Shutting down RAG System...")

app = FastAPI(title="RAG Incident Reporter API", lifespan=lifespan)

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnomalyEvent(BaseModel):
    type: str
    timestamp: str
    location: Optional[str] = "Unknown"
    description: Optional[str] = "Anomaly detected."

class ReportResponse(BaseModel):
    report: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "RAG Incident Report Generator API"}
@app.post("/api/generate", response_model=ReportResponse)
async def generate_report(event: AnomalyEvent):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    try:
        # Convert pydantic model to dict
        event_dict = event.dict()
        if not event_dict.get("location") or not event_dict["location"].strip():
            event_dict["location"] = "Unknown"
        report = await asyncio.to_thread(rag_system.create_report, event_dict)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh-kb")
async def refresh_knowledge_base():
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    try:
        await asyncio.to_thread(rag_system.initialize_knowledge_base)
        return {"status": "success", "message": "Knowledge base re-ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File, Form
import shutil

@app.post("/api/generate-video", response_model=ReportResponse)
async def generate_report_video(
    type: str = Form(...),
    timestamp: str = Form(...),
    location: Optional[str] = Form("Unknown"),
    description: str = Form("Anomaly detected."),
    file: UploadFile = File(...)
):
    """Handles multi-modal incident reporting where users upload a crime video piece of evidence."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG System not initialized")
    
    loc = location.strip() if location and location.strip() else "Unknown"
    event_dict = {
        "type": type,
        "timestamp": timestamp,
        "location": loc,
        "description": description
    }
    
    # Save the file temporarily
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    mp4_converted_path = None
    
    try:
        def _save_file():
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        await asyncio.to_thread(_save_file)
            
        final_video_path = temp_file_path
        
        # Enforce MP4 format using imageio_ffmpeg if the user uploaded a WEBM or MOV
        if temp_file_path.lower().endswith('.webm') or temp_file_path.lower().endswith('.mov'):
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                mp4_converted_path = temp_file_path.rsplit('.', 1)[0] + "_converted.mp4"
                print(f"Transcoding {temp_file_path} to standard MP4...")
                
                def _transcode():
                    import subprocess
                    return subprocess.run([
                        ffmpeg_exe, "-y", "-i", temp_file_path,
                        "-vcodec", "libx264", "-crf", "28", "-preset", "fast",
                        mp4_converted_path
                    ], capture_output=True, text=True)
                
                result = await asyncio.to_thread(_transcode)
                
                if os.path.exists(mp4_converted_path) and os.path.getsize(mp4_converted_path) > 0:
                    final_video_path = mp4_converted_path
                    print("Conversion to MP4 successful!")
                else:
                    print(f"FFmpeg stderr: {result.stderr[-200:]}")
                    raise HTTPException(
                        status_code=422,
                        detail="⚠️ Your video file (.webm/.mov) could not be converted to MP4. Please convert your video to .mp4 using VLC or an online converter and try again."
                    )
            except HTTPException:
                raise
            except Exception as conv_err:
                raise HTTPException(
                    status_code=422,
                    detail=f"⚠️ Video format not supported. Please convert your video to standard .mp4 (H.264) and try again. Error: {str(conv_err)}"
                )

        # Run PyTorch Binary Anomaly Detection
        if pytorch_model:
            print(f"Running real-time PyTorch Inference on {final_video_path}...")
            detected_label, confidence = await asyncio.to_thread(
                predict_anomaly, pytorch_model, final_video_path, pytorch_config, pytorch_device
            )
            if detected_label == "Anomaly":
                print(f"Anomaly confirmed by PyTorch model (Confidence: {confidence:.2f}). Category: '{type}'.")
                event_dict["anomaly_detected"] = True
                event_dict["model_confidence"] = confidence
            else:
                print(f"Model classified video as Normal (Confidence: {confidence:.2f}). Category: '{type}'.")
                event_dict["anomaly_detected"] = False
                event_dict["model_confidence"] = confidence

        # Keep the user's selected incident type as the category
        event_dict["type"] = type
        
        report = await asyncio.to_thread(
            rag_system.create_report_from_video, event_dict, final_video_path
        )
        return {"report": report}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server video processing error: {str(e)}")
    finally:
        # Clean up the large local video file(s) after analysis
        def _cleanup():
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if mp4_converted_path and os.path.exists(mp4_converted_path):
                os.remove(mp4_converted_path)
        await asyncio.to_thread(_cleanup)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
