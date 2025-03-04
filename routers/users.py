# routers/users.py
from fastapi import APIRouter, HTTPException, Request, status
from models import User, UserCreate, UserUpdate, PyObjectId, pwd_context
from typing import List

router = APIRouter()

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(request: Request, user: UserCreate):
    db = request.app.state.db
    # Check for duplicate email
    existing = await db.users.find_one({
        "email": user.email
    })
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    user_data = user.dict()
    user_data["hashed_password"] = pwd_context.hash(user_data.pop("password"))
    result = await db.users.insert_one(user_data)
    new_user = await db.users.find_one({"_id": PyObjectId(result.inserted_id)})
    return new_user

@router.get("/", response_model=List[User])
async def get_users(request: Request):
    db = request.app.state.db
    users = await db.users.find().to_list(100)
    return users

@router.get("/{user_id}", response_model=User)
async def get_user(request: Request, user_id: str):
    db = request.app.state.db
    user = await db.users.find_one({"_id": PyObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}", response_model=User)
async def update_user(request: Request, user_id: str, user_update: UserUpdate):
    db = request.app.state.db
    update_data = {k: v for k, v in user_update.dict().items() if v is not None}
    if "password" in update_data:
        update_data["hashed_password"] = pwd_context.hash(update_data.pop("password"))
    result = await db.users.update_one({"_id": PyObjectId(user_id)}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or no changes made")
    user = await db.users.find_one({"_id": PyObjectId(user_id)})
    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(request: Request, user_id: str):
    db = request.app.state.db
    result = await db.users.delete_one({"_id": PyObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return
