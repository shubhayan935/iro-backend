# models.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from bson import ObjectId
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Helper to validate ObjectId fields
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, info):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}

# ===== Organization models removed =====

# User Models
class UserBase(BaseModel):
    email: EmailStr
    role: str  # "Admin" or "Employee"

class UserCreate(UserBase):
    # When a user is created from the dashboard, default password is "12345678"
    password: str = "12345678"

class UserUpdate(BaseModel):
    email: Optional[EmailStr]
    role: Optional[str]
    password: Optional[str]

class User(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_password: str

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}

# Agent Models
class AgentBase(BaseModel):
    name: str
    role: str  # e.g. "Software Engineer"
    description: Optional[str] = None

class AgentCreate(AgentBase):
    emails: List[EmailStr] = []  # Authorized employee emails
    steps: List[dict] = []        # Onboarding steps

class AgentUpdate(BaseModel):
    name: Optional[str]
    role: Optional[str]
    description: Optional[str]
    emails: Optional[List[EmailStr]]
    steps: Optional[List[dict]]

class Agent(AgentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    emails: List[EmailStr] = []
    steps: List[dict] = []
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}
