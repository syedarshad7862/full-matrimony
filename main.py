from fastapi import FastAPI,Request,Form, File, UploadFile, HTTPException,Depends,Response
from config.db import client,agents_db
from models.user import UserRegister,UserLogin,UserUpdateBioData
from utils import hash_password, verify_password, create_jwt_token,create_access_token, get_authenticated_agent_db
from auth import get_current_user,admin_required
from fastapi.responses import RedirectResponse, StreamingResponse,JSONResponse
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from functions.extract_text import extract_text_from_pdf
from functions.search_matches import create_chunks,extract_indices_from_vector,semantic_search_llm,create_faiss_index,transform_llm_response
import os
import json
import datetime
from bson import ObjectId
from bson.errors import InvalidId
import pdb
from dataclasses import asdict
from typing import Optional
import csv
import io
import time

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
# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

# @app.get("/profiles")
# async def get_profiles(request: Request,user_db=Depends(get_authenticated_agent_db)):
#     user, db = user_db
#     # all_profiles = await db["user_profiles"].find({}).to_list(length=None)
#     # return templates.TemplateResponse("profiles.html", {"request": request, "user": user, 'profiles': all_profiles})
#     return templates.TemplateResponse("profiles.html", {"request": request, "user": user})

@app.get("/edit-profiles/{profile_id}")
async def edit_profile(request: Request,profile_id: str,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    try:
        obj_id = ObjectId(profile_id)
    except InvalidId:
        return {"error": "Invalid ID"}
    profile = await db["user_profiles"].find_one({'_id': obj_id})
    return templates.TemplateResponse("edit_profile.html", {"request": request, "user": user, 'profile':profile})

@app.post("/edit-profile/{profile_id}")
async def edit_profile_db(request: Request,
    profile_id: str,
    full_name: str = Form(...),
    age: str = Form(...),
    date_of_birth: str = Form(...),
    gender: str = Form(...),
    marital_status: str = Form(...),
    complexion: str = Form(...),
    height: str = Form(...),
    education: str = Form(...),
    maslak_sect: str = Form(...),
    occupation: str = Form(...),
    native_place: str = Form(...),
    residence: str = Form(...),
    location: str = Form(...),
    siblings: str = Form(...),
    father_name: str = Form(...),
    mother_name: str = Form(...),
    preferences: str = Form(...),
    pref_age_range: str = Form(...),
    pref_marital_status: str = Form(...),
    pref_height: str = Form(...),
    pref_complexion: str = Form(...),
    pref_education: str = Form(...),
    pref_work_job: str = Form(...),
    pref_father_occupation: str = Form(...),
    pref_no_of_siblings: str = Form(...),
    pref_native_place: str = Form(...),
    pref_mother_tongue: str = Form(...),
    pref_go_to_dargah: str = Form(...),
    pref_maslak_sect: str = Form(...),
    pref_deendari: str = Form(...),
    pref_location: str = Form(...),
    pref_own_house: str = Form(...),
    user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    try:
        obj_id = ObjectId(profile_id)
    except InvalidId:
        return {"error": "Invalid ID"}
    updated_data = {
        "full_name": full_name,
        "age": age,
        "date_of_birth": date_of_birth,
        "gender": gender,
        "marital_status": marital_status,
        "complexion": complexion,
        "height": height,
        "education": education,
        "maslak_sect": maslak_sect,
        "occupation": occupation,
        "native_place": native_place,
        "residence": residence,
        "location": location,
        "siblings": siblings,
        "father_name": father_name,
        "mother_name": mother_name,
        "preferences": preferences,
        "pref_age_range": pref_age_range,
        "pref_marital_status": pref_marital_status,
        "pref_height": pref_height,
        "pref_complexion": pref_complexion,
        "pref_education": pref_education,
        "pref_work_job": pref_work_job,
        "pref_father_occupation": pref_father_occupation,
        "pref_no_of_siblings": pref_no_of_siblings,
        "pref_native_place": pref_native_place,
        "pref_mother_tongue": pref_mother_tongue,
        "pref_go_to_dargah": pref_go_to_dargah,
        "pref_maslak_sect": pref_maslak_sect,
        "pref_deendari": pref_deendari,
        "pref_location": pref_location,
        "pref_own_house": pref_own_house,
    }
    result = await db['user_profiles'].update_one(
        {"_id": obj_id},
        {"$set": updated_data}
    )
    print(result)
    # return templates.TemplateResponse("edit_profile.html", {"request": request, "user": user})
    return RedirectResponse(url="/profiles", status_code=303)

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
    # print(full_name)
    # print(f"top_k value: {top} type: {type(top)}")
    print(db.name)
    
    # function for dataframe & chunks
    texts ,profile_df = await create_chunks(MONGO_URI,db.name,"user_profiles")
    
    matched_profiles, query_text = extract_indices_from_vector(profile_df,full_name,top)
    # print(matched_profiles.columns)
    # if 'text' in matched_profiles.columns:
    #     print(f"profile_df: {matched_profiles}\n query_text: {query_text} \n {matched_profiles['text'].tolist()}")
    # else:
    #     print(f"❌ 'text' column not found! Available columns: {matched_profiles.columns}")

    if matched_profiles.empty:
        print("No Matches found!")
    else:
        llm_response = semantic_search_llm(matched_profiles, query_text)
    print(llm_response)
    result = transform_llm_response(llm_response)
    # Convert all Match dataclass objects to plain dictionaries
    matches = [m.dict() for m in result['matches']]
    print(f"selected_profile: {selected_profile} \n\n Matches Profiles: {matches} \n\n matched_profiles: {matched_profiles}")
    # pdb.set_trace()
    return templates.TemplateResponse(
    "find_matches.html",
    {
        "request": request,
        "user": user,
        "profiles": await db["user_profiles"].find({}).to_list(length=None),
        "selected_profile": selected_profile,
        "matched_profiles": matches,  # your transformed matches
        "top": top
        
    }
)

@app.post("/download_matches_csv")
async def download_matches_csv(
    profile_id: str = Form(...),
    top: str = Form(...),
    user_db=Depends(get_authenticated_agent_db)
):
    user, db = user_db
    print(f"{top} type of top: {type(top)}")
    selected_profile = await db["user_profiles"].find_one({"_id": ObjectId(profile_id)})
    if not selected_profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    full_name = selected_profile.get("full_name", "Name not found")

    texts, profile_df = await create_chunks(MONGO_URI, db.name, "user_profiles")

    matched_profiles, query_text = extract_indices_from_vector(profile_df, full_name, int(top))

    if matched_profiles.empty:
        raise HTTPException(status_code=404, detail="No matches found")

    llm_response = semantic_search_llm(matched_profiles, query_text)
    result = transform_llm_response(llm_response)
    matches = [m.dict() for m in result['matches']]
    print(f"matched_profiles: {type(matched_profiles)}\n\n matches: {type(matches)}")
    # pdb.set_trace()
    # Prepare CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Write Selected Profile First
    writer.writerow(["Selected Profile"])  # Title
    for key, value in selected_profile.items():
        writer.writerow([key, value])

    # Optional separator
    writer.writerow([])
    writer.writerow(["Matched Profiles"])

    # if matches:
    #     # Write Matched Profiles Headers
    #     writer.writerow(matches[0].keys())

    #     # Write Matched Profiles Data
    #     for match in matches:
    #         writer.writerow(match.values())
    if not matched_profiles.empty:
        output.write(matched_profiles.to_csv(index=False))

    output.seek(0)

    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=matched_profiles.csv"
    })    
@app.post("/create-vectors")
async def create_vector(request: Request, user_db= Depends(get_authenticated_agent_db)):
    user, db = user_db
    await create_faiss_index(MONGO_URI,db.name,"user_profiles")
    print(MONGO_URI,db.name,"user_profiles")
    return {"message": "Created vectors successfully!"}

# profile completion filter
@app.get("/api/filtered-profiles")
async def profile_completion(request: Request,user_db = Depends(get_authenticated_agent_db), min_completion: Optional[int] = 10):
    try:
        # time.sleep(5)  # simulate delay
        user, db = user_db
        query = {"profile_completion": {"$gte": min_completion}}
        print(query)
        profiles = await db["user_profiles"].find(query).to_list(length=None)
        print(len(profiles))
        # return {"message": profiles}
        return templates.TemplateResponse("profiles.html", {
            "request": request,
            "user": user,
            "total_profiles": len(profiles),
            "profiles": profiles,
            "min_completion": min_completion
        })
        # return JSONResponse(content={"profiles": profiles,"user": user})
    except Exception as e:
        return HTTPException(status_code=500, detail="Error: {e}")

@app.get("/api/full-details")
async def full_details_json(profile_id: int, user_db = Depends(get_authenticated_agent_db)):
    user, db = user_db
    profile = await db["user_profiles"].find_one({"profile_id": profile_id})
    if not profile:
        return JSONResponse(status_code=404, content={"error": "Profile not found"})
    # Convert ObjectId to str
    profile["_id"] = str(profile["_id"])
    print(profile)
    return profile  # FastAPI will return JSON

# @app.post("/api/full-details")
# async def full_details(request: Request, profile_id: int = Form(...), user_db = Depends(get_authenticated_agent_db)):
#     user, db = user_db
#     profile = await db["user_profiles"].find_one({"profile_id": profile_id})
#     return templates.TemplateResponse("full_profile.html", {
#         "request": request,
#         "profile": profile
#     })
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