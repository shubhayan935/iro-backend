# main.py
import os
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from routers import users, agents, auth
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import certifi

load_dotenv()  # Load environment variables from a .env file

app = FastAPI(title="Iro Onboarding Backend")

# MongoDB connection – set MONGODB_URL in your .env file or use the default below.
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
print(f"Connecting to MongoDB at {MONGODB_URL}...")
client = AsyncIOMotorClient(
    MONGODB_URL,
    tlsCAFile=certifi.where()
)
db = client["iro"]  # Database name
print("Connected to MongoDB. Using database 'iro'.")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach the DB to the app state so that routers can access it
app.state.db = db

# Include our routers (organizations router and seed endpoint removed)
app.include_router(users.router, prefix="/users", tags=["Users"])
print("Users router included at '/users/'")
app.include_router(agents.router, prefix="/agents", tags=["Agents"])
print("Agents router included at '/agents/'")
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
print("Auth router included at '/auth/'")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
