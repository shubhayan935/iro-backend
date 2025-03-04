# routers/auth.py
from fastapi import APIRouter, HTTPException, Request, status
from models import User, PyObjectId, pwd_context
from pydantic import BaseModel

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login", response_model=User)
async def login(request: Request, credentials: LoginRequest):
    db = request.app.state.db
    user = await db.users.find_one({
        "email": credentials.email
    })
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Please contact your admin.")
    if not pwd_context.verify(credentials.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return user
