"""
FastAPI server for Athena Task - Shark Tank AI Pitch Analyzer
Provides REST API endpoints for file upload and pitch analysis
"""

import logging
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------------
# Ensure project root is on sys.path so `backend` package is importable
# This allows running the server directly from the `backend` directory:
#   python api_server.py
# and also from the project root as:
#   python -m backend.api_server
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import pipeline functions from backend package
from backend.main import run_pipeline, validate_input_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose watchfiles logging (only show warnings and errors)
logging.getLogger('watchfiles.main').setLevel(logging.WARNING)

# Verify environment variables are loaded
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables. Make sure .env file is configured.")
if GROQ_KEY:
    logger.info("Groq API key loaded successfully from environment variables.")

# Initialize FastAPI app
app = FastAPI(
    title="Athena Task API",
    description="Shark Tank AI Pitch Analyzer API",
    version="1.0.0"
)

# Configure CORS - Allow all localhost origins in development
# For production, replace with specific allowed origins
def is_localhost_origin(origin: str) -> bool:
    """Check if origin is localhost (for development)"""
    if not origin:
        return False
    return origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # Allow any localhost port
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (default)
        "http://localhost:8080",  # Vite dev server (custom port)
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Response models
class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    file_path: str
    message: str

class StartAnalysisRequest(BaseModel):
    file_path: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Ensure upload directory exists
UPLOAD_DIR = ROOT_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage for analysis results (in production, use Redis or database)
analysis_results: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Athena Task API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is operational"
    )

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and store it for later analysis"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    supported_extensions = ['.mp4', '.avi', '.mov', '.wav', '.mp3', '.mpeg', '.mpg']
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}"
        )
    
    # Use a safe filename or generate one
    safe_filename = f"{tempfile.mktemp(dir='')}_{file.filename}"
    target_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded to: {target_path}")
        return UploadResponse(
            success=True,
            file_path=str(target_path),
            message="File uploaded successfully"
        )
    except Exception as e:
        logger.exception(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/start-analysis", response_model=AnalysisResponse)
async def start_analysis(request: StartAnalysisRequest):
    """Start analysis on a previously uploaded file"""
    file_path = Path(request.file_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.info(f"Starting analysis for: {file_path}")
    
    try:
        result = run_pipeline(str(file_path))
        logger.info("Analysis pipeline completed successfully")
        
        response_data = {
            "voice_stats": result.get("voice_stats", {}),
            "transcript": result.get("transcript", ""),
            "transcript_segments": result.get("transcript_segments", []),
            "content_analysis": result.get("content_analysis", {}),
            "shark_panel": result.get("shark_panel", {}),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
        }
        
        # Optionally clean up the file after analysis
        # os.remove(file_path)
        
        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            data=response_data
        )
    except Exception as e:
        logger.exception(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_pitch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and analyze a pitch file (video or audio)
    
    Args:
        file: Uploaded media file (MP4, AVI, MOV, WAV, MP3)
        
    Returns:
        Analysis results including voice stats, content analysis, and shark panel feedback
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    supported_extensions = ['.mp4', '.avi', '.mov', '.wav', '.mp3', '.mpeg', '.mpg']
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(supported_extensions)}"
        )
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file
        logger.info(f"Receiving file: {file.filename} ({file.size} bytes)")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to: {temp_file_path}")
        
        # Validate file exists and is readable
        if not os.path.exists(temp_file_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Run analysis pipeline
        logger.info("Starting analysis pipeline...")
        try:
            result = run_pipeline(temp_file_path)
            logger.info("Analysis pipeline completed successfully")
            
            # Format response data
            response_data = {
                "voice_stats": result.get("voice_stats", {}),
                "transcript": result.get("transcript", ""),
                "transcript_segments": result.get("transcript_segments", []),
                "content_analysis": result.get("content_analysis", {}),
                "shark_panel": result.get("shark_panel", {}),
                "file_name": file.filename,
                "file_size": file.size,
            }
            
            return AnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                data=response_data
            )
            
        except Exception as e:
            logger.exception(f"Analysis pipeline failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {str(e)}")

@app.get("/api/status/{job_id}")
async def get_analysis_status(job_id: str):
    """Get status of an analysis job (for async operations)"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return analysis_results[job_id]

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

