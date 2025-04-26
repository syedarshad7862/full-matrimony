import bcrypt
import jwt 
import os
from datetime import datetime, timedelta,timezone
from dotenv import load_dotenv
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import Request, HTTPException,Depends
from fastapi.responses import RedirectResponse
from typing import Optional


# security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login_agent")
load_dotenv()
SECRET_KEY = "matrimonial-meer-ahmed-sir"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

MONGO_URI = os.getenv("MONGO_URI")
# connect = MongoClient(MONGO_URI)
client = AsyncIOMotorClient(MONGO_URI)
agents_db = client["matrimony_agents"] # Common DB for storing agents
# user_collection = db["user"]

# def hash_password(password: str) -> str:
#     salt = bcrypt.gensalt()
#     return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

# def verify_password(password: str, hashed_password: str) -> bool:
#     return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
# Helper Functions

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    return pwd_context.hash(password)

def create_jwt_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_jwt_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        print("Token expired")
        return None
    except jwt.InvalidTokenError:
        print("inviled Token expired")
        return None
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})  # 🔄 Expiration claim
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)  # 🔑 Encode JWT
# def get_authenticated_agent_db(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#         agent_id = payload["agent_id"]
#         agent_username = payload["agent_username"]
#         return client[f"matrimony_{agent_username}_{agent_id}"]  # Return the agent's personal DB
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token expired")
#     except jwt.InvalidTokenError:
#         raise HTTPException(status_code=401, detail="Invalid token")
# def get_authenticated_agent_db(request: Request):
#     token = request.cookies.get("access_token")
#     if not token:
#         raise HTTPException(status_code=401, detail="Not authenticated")  # 🔴 Fix: Return proper error if missing token

#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#         agent_id = payload["agent_id"]
#         agent_username = payload["agent_username"].lower()
#         return client[f"matrimony_{agent_username}_{agent_id}"]
#     except jwt.ExpiredSignatureError:
#         # raise HTTPException(status_code=401, detail="Token expired")
#         return RedirectResponse(url="/", status_code=303)
#     except jwt.InvalidTokenError:
#         # raise HTTPException(status_code=401, detail="Invalid token")
#         return RedirectResponse(url="/", status_code=303)

def get_authenticated_agent_db(request: Request):
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        agent_id = payload["agent_id"]
        agent_username = payload["agent_username"].lower()  # Always use lowercase to avoid case issues
        user = {"agent_id": agent_id, "agent_username": agent_username, "email": payload["email"]}
        agent_db = client[f"matrimony_{agent_username}_{agent_id}"]
        return user, agent_db # ✅ Return agent's DB
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
        # return RedirectResponse(url="/", status_code=303)  # 🔄 Redirect if token expired
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
        # return RedirectResponse(url="/", status_code=303)  # 🔄 Redirect if token is invalid




