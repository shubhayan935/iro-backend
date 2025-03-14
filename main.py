# main.py
import os
import logging
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from routers import users, agents, auth, recordings
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import certifi
from contextlib import asynccontextmanager
import models_loader
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix SSL verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application...")
    
    # Load models at startup
    logger.info("Loading ML models at startup...")
    try:
        await models_loader.load_models_async()
        logger.info("Models loaded successfully at startup")
    except Exception as e:
        logger.error(f"Error loading models at startup: {e}")
        logger.info("Application will continue without some models")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI app with proper lifespan
app = FastAPI(
    title="Iro Onboarding Backend",
    description="API for onboarding process management",
    lifespan=lifespan
)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
logger.info(f"Connecting to MongoDB at {MONGODB_URL}...")

try:
    client = AsyncIOMotorClient(
        MONGODB_URL,
        tlsCAFile=certifi.where(),
    )
    # Verify connection works
    client.admin.command('ping')
    logger.info("MongoDB connection successful")
    
    db = client["iro"]
    # Attach the DB to the app state
    app.state.db = db
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    # Continue anyway - some routes might not need the database

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    model_status = models_loader.get_model_status()
    return {
        "status": "healthy",
        "models": model_status
    }

# Include routers
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(agents.router, prefix="/agents", tags=["Agents"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(recordings.router, prefix="/recordings", tags=["Recordings"])

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info",
            ssl_keyfile=None,  # Disable SSL for local development
            ssl_certfile=None  # Disable SSL for local development
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port 8000 is already in use. Try killing the existing process.")
            # Suggest a different port
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            available_port = s.getsockname()[1]
            s.close()
            logger.info(f"Try using port {available_port} instead by running:")
            logger.info(f"python -m uvicorn main:app --host 0.0.0.0 --port {available_port}")
        else:
            logger.error(f"Error starting server: {e}")