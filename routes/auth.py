
from fastapi import FastAPI, APIRouter, Form,Request,HTTPException,Depends,Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import datetime
import os
from passlib.context import CryptContext
from config.db import client,agents_db
from models.user import UserRegister,UserLogin,UserUpdateBioData
from utils import hash_password, verify_password, create_jwt_token,create_access_token, get_authenticated_agent_db
from auth import get_current_user,admin_required
from starlette.responses import HTMLResponse
app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# database
MONGO_URI = os.getenv("MONGO_URI")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix='/auth', tags=['Auth'])

@router.get("/register")
def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register_agent/")
async def register_agent(
    request: Request,
    response: Response,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    ):
    existing_user = await agents_db["agents"].find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(password)
    agent_data = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }
    result = await agents_db["agents"].insert_one(agent_data)
    
    agent_id = str(result.inserted_id)
    
    # Create a separate database for this agent
    agent_db_name = f"matrimony_{username.lower().replace(' ', '_')}_{agent_id}"
    agent_db = client[agent_db_name]
    await agent_db["user_profiles"].insert_one({"message": "Agent DB Initialized"})

    # return {"message": "Agent registered successfully", "agent_id": agent_id, "agent_db_name": agent_db_name}
    response = RedirectResponse("/login", status_code=303)
    return response


@router.get("/login")
def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...)
    ):
        # Validate form data using Pydantic
    try:
        login_data = UserLogin(email=email, password=password)
    except Exception as e:
        return templates.TemplateResponse(
            "login.html", {"request": request, "message": f"Invalid input: {str(e)}"}
        )
    
    # Find user in database
    db_agents = await agents_db["agents"].find_one({"email": email})
    if not db_agents or not verify_password(login_data.password, db_agents["password"]):
        return templates.TemplateResponse(
            "login.html", {"request": request, "message": "Invalid email or password"}
        )
    # Check if user already added biodata
    # if "biodata" in db_agents and db_agents["biodata"]:
    #     redirect_url = "/match"  # Redirect to match.html if biodata exists
    # else:
    #     redirect_url = "/add-bio"  # Redirect to add-bio if biodata is missing
    # Generate JWT token
    access_token = create_access_token({"agent_id": str(db_agents["_id"]),"agent_username":str(db_agents["username"]), "email": db_agents["email"]})

    # Set token as an HTTP-only cookie
    response = RedirectResponse("/auth/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True)
    return response

# logout route
@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")  # Remove token
    return response


# only login user access
@router.get("/dashboard", response_class=HTMLResponse)
async def agent_dashboard(request: Request,user_db=Depends(get_authenticated_agent_db)):
    # try:
    #     user = get_current_user(request)  # 🔄 Get the authenticated user
    #     print(total_profiles)
    # except HTTPException as e:
    #     if e.status_code == 401:  # ⛔ If token expired or invalid, redirect to home
    #         return RedirectResponse(url="/", status_code=303)
    # return RedirectResponse(url="/", status_code=303)
    user, db = user_db  # Unpack user & database from function
    total_profiles = await db["user_profiles"].count_documents({})
    
    # Count male users
    total_male = await db["user_profiles"].count_documents({"gender": "Male"})

    # Count female users
    total_female = await db["user_profiles"].count_documents({"gender": "Female"})
    print(user)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user,"total_profiles": total_profiles,"total_male": total_male,"total_female": total_female,})