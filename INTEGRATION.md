# Backend-Frontend Integration Guide

This document explains how to run and integrate the backend API server with the frontend React application.

## Architecture Overview

- **Backend**: FastAPI server running on `http://localhost:8000`
- **Frontend**: React + Vite application running on `http://localhost:5173` (or port 8080)

## Setup Instructions

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create `.env` file in `backend/`):
   ```env
   GROQ_API_KEY=your_groq_api_key
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

4. Start the API server:
   ```bash
   python api_server.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`

### 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   bun install
   # or
   npm install
   ```

3. (Optional) Set API URL in `.env` file:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. Start the development server:
   ```bash
   bun run dev
   # or
   npm run dev
   ```

The frontend will be available at `http://localhost:5173` (or the port shown in terminal)

## API Endpoints

### POST `/api/analyze`
Upload and analyze a pitch file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (video/audio file)

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "data": {
    "voice_stats": { ... },
    "transcript": "...",
    "content_analysis": { ... },
    "shark_panel": { ... }
  }
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is operational"
}
```

## Data Flow

1. User uploads a file through the frontend
2. Frontend sends file to `/api/analyze` endpoint
3. Backend processes the file through the analysis pipeline:
   - Audio processing
   - Voice analysis
   - Speech-to-text (ASR)
   - Content analysis
   - Shark panel evaluation
4. Backend returns JSON response with all analysis results
5. Frontend displays results in:
   - VoiceAnalysisCard (voice metrics, delivery score, emotional tone)
   - ContentAnalysisCard (business viability, pitch structure, transcript)
   - SharkPanel (individual shark feedback and verdicts)

## Troubleshooting

### CORS Errors
If you see CORS errors, ensure:
- Backend CORS middleware is configured correctly
- Frontend is accessing the correct API URL
- Both servers are running

### File Upload Issues
- Check file size limits (may need to adjust in FastAPI)
- Verify file format is supported (MP4, AVI, MOV, WAV, MP3)
- Check backend logs for detailed error messages

### API Connection Issues
- Verify backend is running on port 8000
- Check `VITE_API_URL` environment variable in frontend
- Test API health endpoint: `curl http://localhost:8000/health`

## Development Notes

- Backend logs are written to `backend/api_server.log`
- Frontend uses React Query for state management (if implemented)
- File uploads are temporarily stored during processing
- Analysis results are returned synchronously (consider async for large files)

## Production Deployment

For production:
1. Set proper CORS origins
2. Use environment variables for all secrets
3. Implement file size limits
4. Add rate limiting
5. Use a proper file storage solution (S3, etc.)
6. Consider async processing with job queues for large files

