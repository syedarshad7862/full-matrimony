from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request, HTTPException,Depends
from utils import decode_jwt_token
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import RedirectResponse

security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login_agent")
import jwt

SECRET_KEY = "matrimonial-meer-ahmed-sir"
ALGORITHM = "HS256"
# def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
#     token = credentials.credentials
#     print(f"Received Token: {token}")
#     payload = decode_jwt_token(token)
#     if not payload:
#         raise HTTPException(status_code=401, detail="Invalid or expired token")
#     return payload
# def get_current_user(request: Request):
#     token = request.cookies.get("access_token")  # Extract from cookies
#     if not token:
#         raise HTTPException(status_code=401, detail="Not authenticated")

#     payload = decode_jwt_token(token)
#     if not payload:
#         raise HTTPException(status_code=401, detail="Invalid or expired token")

#     return payload
def get_current_user(request: Request):
    token = request.cookies.get("access_token")  # Extract token from cookies

    if not token:
        print("No token found. Sending 401 response.")
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_jwt_token(token)

    if not payload:  # If payload is None, token is invalid or expired
        print("JWT token has expired or is invalid. Sending 401 response.")
        raise HTTPException(status_code=401, detail="Token expired")  # ⛔ Return 401

    return payload  # ✅ Return user data if token is validn payload  # ✅ Return user data if token is valid
def admin_required(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


