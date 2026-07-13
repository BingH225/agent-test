"""
Smart Stress Agent - FastAPI Server
Serves both API endpoints and frontend UI
"""
import uvicorn
import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from smartstress_langgraph.io_models import ContinueSessionRequest, StartSessionRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Stress Agent API",
    description="AI-powered stress monitoring and intervention system",
    version="1.0.0"
)

# CORS Configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check Endpoint ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smart Stress Agent API",
        "version": "1.0.0",
        "physio_model_id": "wesad_attention_v1",
        "task_relief_mode": "dry_run",
    }

# --- API Endpoints ---

@app.post("/api/start_session")
async def api_start_session(req: StartSessionRequest):
    """
    Start a new monitoring session
    
    Args:
        req: StartSessionRequest data
        
    Returns:
        Session handle and initial state view
    """
    try:
        # Import here to catch import errors gracefully
        from smartstress_langgraph.api import start_monitoring_session
        handle, view = start_monitoring_session(req)
        
        logger.info(f"Started session for user: {req.user.user_id}")
        
        # Convert Pydantic models to dict
        return {
            "success": True,
            "handle": handle.model_dump(),
            "view": view.model_dump()
        }
    except ImportError as e:
        logger.error(f"Import error in start_session: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server configuration error: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in start_session: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start session: {str(e)}"
        )

@app.post("/api/continue_session")
async def api_continue_session(req: ContinueSessionRequest):
    """
    Continue an existing monitoring session
    
    Args:
        req: ContinueSessionRequest data
        
    Returns:
        Updated session handle and state view
    """
    try:
        # Import here to catch import errors gracefully
        from smartstress_langgraph.api import continue_session
        handle, view = continue_session(req)
        
        logger.info(f"Continued session: {req.session_handle.thread_id}")
        
        return {
            "success": True,
            "handle": handle.model_dump(),
            "view": view.model_dump()
        }
    except ImportError as e:
        logger.error(f"Import error in continue_session: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server configuration error: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in continue_session: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to continue session: {str(e)}"
        )

# --- Frontend Integration ---

# Path to the external frontend build directory
# Path to the external frontend build directory
# Must be set via FRONTEND_PATH environment variable
env_frontend_path = os.getenv("FRONTEND_PATH")
if not env_frontend_path:
    # Default to a safe relative path or raise warning/error depending on preference
    # Here we default to a local 'frontend_dist' folder to avoid leaking absolute paths
    FRONTEND_DIST = Path("frontend_dist")
    logger.warning("FRONTEND_PATH not set in .env. Defaulting to ./frontend_dist")
else:
    FRONTEND_DIST = Path(env_frontend_path)

# Check if frontend exists
frontend_available = FRONTEND_DIST.exists() and FRONTEND_DIST.is_dir()

if frontend_available:
    logger.info(f"Frontend directory found at: {FRONTEND_DIST}")
    
    # Check for index.html
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        logger.warning(f"index.html not found in {FRONTEND_DIST}")
        frontend_available = False
    else:
        logger.info(f"Frontend index.html found")
    
    # Mount assets if they exist
    assets_path = FRONTEND_DIST / "assets"
    if assets_path.exists() and assets_path.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")
        logger.info(f"Mounted assets directory")
    
    # Serve index.html for root and any other path (SPA fallback)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """
        Serve the frontend SPA
        Falls back to index.html for client-side routing
        """
        # Don't intercept API calls
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Check if it's a file request that exists in dist
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        # Otherwise serve index.html for SPA routing
        index_path = FRONTEND_DIST / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        else:
            logger.error("index.html not found when serving SPA")
            return JSONResponse(
                {"error": "Frontend index.html not found"}, 
                status_code=404
            )
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIST}")
    logger.warning("Server will run in API-only mode")
    
    @app.get("/")
    async def root():
        """Root endpoint when frontend is not available"""
        return {
            "status": "running",
            "message": "Smart Stress Agent API is running",
            "note": "Frontend UI not found - API-only mode",
            "frontend_path": str(FRONTEND_DIST),
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "api": {
                    "start_session": "/api/start_session",
                    "continue_session": "/api/continue_session"
                }
            }
        }

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("Smart Stress Agent Server Starting")
    logger.info("=" * 60)
    logger.info(f"Frontend available: {frontend_available}")
    if frontend_available:
        logger.info(f"Frontend path: {FRONTEND_DIST}")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("=" * 60)

# --- Main Entry Point ---

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Smart Stress Agent Server")
    print("=" * 60)
    print(f"Frontend directory: {FRONTEND_DIST}")
    print(f"Frontend available: {frontend_available}")
    print("\nServer will start on: http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
