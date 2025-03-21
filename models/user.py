from pydantic import BaseModel, EmailStr,Field
from typing import Optional

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "agent"  # Default role is "user"

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    
class UserBioData(BaseModel):
    full_name: str = Field(..., min_length=3, max_length=50)
    age: int = Field(..., gt=18, lt=100)
    gender: str 
    education: str
    profession: str
    location: str
    preference: str 
    
class UserUpdateBioData(BaseModel):
    full_name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    education: Optional[str]
    profession: Optional[str]
    location: Optional[str]
    preference: Optional[str]
    