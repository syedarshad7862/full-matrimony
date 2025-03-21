from fastapi import FastAPI,Request,Form, File, UploadFile, HTTPException,Depends,Response
from config.db import client,agents_db
from models.user import UserRegister,UserLogin,UserUpdateBioData
from utils import hash_password, verify_password, create_jwt_token,create_access_token, get_authenticated_agent_db
from auth import get_current_user,admin_required
from fastapi.responses import RedirectResponse
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from functions.extract_text import extract_text_from_pdf
from functions.search_matches import create_chunks,extract_indices_from_vector,semantic_search_llm,create_faiss_index,transform_llm_response
import os
import json
import datetime
from bson import ObjectId
import pdb
app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
UPLOAD_DIR = "uploads"  # Directory to store uploaded PDFs
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create folder if not exists
# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# # Add the middleware
# app.add_middleware(AuthMiddleware)
# home route
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
# @app.get("/client")
# async def home(request: Request, user: dict = Depends(get_current_user)):
#     db_agents = await user_collection.find_one({"User-Credentials.email": user["sub"]})
#     username = db_agents["User-Credentials"]["username"]
#     # Check if user already added biodata
#     if  db_agents and "biodata" in db_agents:
#         href_url = "/edit-bio"  # sending url to frontend
#         title = "Edit-Profile"
#         title_2 = "Delete-Profile"
#         href_delete = "/delete"
#     else:
#         href_url = "/add-bio"  # Redirect to add-bio if biodata is missing
#         title = "Add-Profile"
#         title_2 = "Upload_Pdf"
#         href_delete = "/upload"
#     return templates.TemplateResponse("client.html", {"request": request, "user": user,"href_url":href_url,"title":title, "username": username,"href_delete":href_delete,"title_2":title_2})
# @app.get("/upload")
# async def upload_pdf(request: Request, user: dict = Depends(get_current_user)):
#     db_agents = await user_collection.find_one({"User-Credentials.email": user["sub"]})
    
#     # Check if user already added biodata
#     if  db_agents and "biodata" in db_agents:
#         href_url = "/edit-bio"  # sending url to frontend
#         title = "Edit-Profile"
#         title_2 = "Delete-Profile"
#         href_delete = "/delete"
#         username = db_agents["User-Credentials"]["username"]
#     else:
#         href_url = "/add-bio"  # Redirect to add-bio if biodata is missing
#         title = "Add-Profile"
#         title_2 = "Upload_Pdf"
#         href_delete = "/upload"
#         username = ''
#     return templates.TemplateResponse("upload.html", {"request": request, "user": user,"href_url":href_url,"title":title, "username": username,"href_delete":href_delete,"title_2":title_2})

@app.get("/register")
def get_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login")
def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/register_agent/")
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

@app.post("/login")
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
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=True)
    return response

from pymongo.errors import PyMongoError
from fastapi import HTTPException

@app.post("/add-profile")
async def add_profile(
    request: Request,
    full_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    marital_status: str = Form(...),
    complexion: str = Form(...),
    height: str = Form(...),
    education: str = Form(...),
    maslak_sect: str = Form(...),
    occupation: str = Form(...),
    native_place: str = Form(...),
    residence: str = Form(...),
    siblings: str = Form(...),
    father: str = Form(...),
    mother: str = Form(...),
    preferences: str = Form(...),
    user_db=Depends(get_authenticated_agent_db)
):
    try:
        user, db = user_db  # Unpack user & database from function
        
        previous_id = await db["user_profiles"].count_documents({})
        # biodata["profile_id"] = previous_id+1
        biodata = {
            "profile_id": previous_id+1,
            "full_name": full_name,
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "complexion": complexion,
            "height": height,
            "education": education,
            "maslak_sect": maslak_sect,
            "occupation": occupation,
            "native_place": native_place,
            "residence": residence,
            "siblings": siblings,
            "father": father,
            "mother": mother,
            "preferences": preferences,
        }

    
        # Debugging logs
        # Attempt to insert data into the database
        added = await db["user_profiles"].insert_one(biodata)
        
        print(previous_id)
        if not added.inserted_id:
            raise HTTPException(status_code=500, detail="Failed to add profile to the database")

        # Redirect to a success page to avoid duplicate submission on refresh
        return RedirectResponse(url="/add-profile", status_code=303)

    except PyMongoError as e:
        print(f"Database Error: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="Database error occurred while adding profile")

    except Exception as e:
        print(f"Unexpected Error: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


# only login user access
@app.get("/dashboard", response_class=HTMLResponse)
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

@app.get("/upload")
def upload_pdf(request: Request,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})   

# upload pdf
@app.post("/upload-pdf")
async def upload_pdf_db(request: Request ,file: UploadFile = File(...), user_db=Depends(get_authenticated_agent_db)):
    file_location = f'{UPLOAD_DIR}/{file.filename}'
    # with open(file_location, "wb") as buffer:
    #     file_contants = await file.read()
    #     print(file_contants)
    #     buffer.write(file_contants)
        
    extracted_text = extract_text_from_pdf(file)
    user, db = user_db
    previous_id = await db["user_profiles"].count_documents({})
    format_data = {
        "profile_id": previous_id+1,
        "unstracture": extracted_text
        }
    added = await db["user_profiles"].insert_one(format_data)
    return RedirectResponse("/upload", status_code=303)

@app.get("/find")
async def find_matches(request: Request,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    all_profiles = await db["user_profiles"].find({}).to_list(length=None)
    # print(all_profiles)
    return templates.TemplateResponse("find_matches.html", {"request": request, "user": user, "profiles": all_profiles,"selected_profile": None})   

@app.post("/show_matches", response_class=HTMLResponse)
async def show_matches(
    request: Request,
    profile_id: str = Form(...),
    top: int = Form(...),
    user_db=Depends(get_authenticated_agent_db)
):
    user, db = user_db

    # Find the selected user profile
    selected_profile = await db["user_profiles"].find_one({"_id": ObjectId(profile_id)})

    if not selected_profile:
        return templates.TemplateResponse(
            "find_matches.html",
            {"request": request, "user": user, "profiles": [], "error": "Profile not found!","selected_profile": selected_profile}
        )
    full_name = selected_profile.get("full_name", "Name not found")
    print(full_name)
    print(f"top_k value: {top} type: {type(top)}")
    print(db.name)
    
    # function for dataframe & chunks
    texts ,profile_df = await create_chunks(MONGO_URI,db.name,"user_profiles")
    
    matched_profiles, query_text = extract_indices_from_vector(profile_df,full_name,top)
    # pdb.set_trace()
    # print(matched_profiles.columns)
    if 'text' in matched_profiles.columns:
        print(f"profile_df: {matched_profiles}\n query_text: {query_text} \n {matched_profiles['text'].tolist()}")
    else:
        print(f"❌ 'text' column not found! Available columns: {matched_profiles.columns}")

    if matched_profiles.empty:
        print("No Matches found!")
    else:
        llm_response = semantic_search_llm(matched_profiles, query_text)
    print(llm_response)
    final_output = transform_llm_response(llm_response)
    print(final_output)
    return RedirectResponse("/find", status_code=303)


    # return templates.TemplateResponse(
    #     "show_matches.html",
    #     {"request": request, "user": user, "selected_profile": selected_profile}
    # )
    
@app.post("/create-vectors")
async def create_vector(request: Request, user_db= Depends(get_authenticated_agent_db)):
    user, db = user_db
    await create_faiss_index(MONGO_URI,db.name,"user_profiles")
    print(MONGO_URI,db.name,"user_profiles")
    return {"message": "Created vectors successfully!"}
# logout route
@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")  # Remove token
    return response
# form route
@app.get("/add-profile")
async def get_profile(request: Request,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db  # Unpack user & database from function
    previous_id = await db["user_profiles"].count_documents({})
        # biodata["profile_id"] = previous_id+1
    print(previous_id)
    return templates.TemplateResponse("add_bio.html", {"request": request, "user": user})

# only login user access
# @app.get("/dashboard")
# def dashboard(user: dict = Depends(get_current_user)):
#     return {"message": "Welcome to user dashboard", "user": user}

@app.get("/admin/dashboard")
def admin_dashboard(admin: dict = Depends(admin_required)):
    return {"message": "Welcome to admin dashboard", "admin": admin}