from fastapi import FastAPI,Request,Depends
from utils import get_authenticated_agent_db
from auth import admin_required
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
# from functions.extract_text import extract_text_from_pdf
import os
from routes import auth, profile, match


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
# route
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(match.router)
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




@app.get("/upload")
def upload_pdf(request: Request,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})      

# upload pdf
# @app.post("/upload-pdf")
# async def upload_pdf_db(request: Request ,file: UploadFile = File(...), user_db=Depends(get_authenticated_agent_db)):
#     file_location = f'{UPLOAD_DIR}/{file.filename}'
#     # with open(file_location, "wb") as buffer:
#     #     file_contants = await file.read()
#     #     print(file_contants)
#     #     buffer.write(file_contants)
        
#     extracted_text = extract_text_from_pdf(file)
#     user, db = user_db
#     previous_id = await db["user_profiles"].count_documents({})
#     format_data = {
#         "profile_id": previous_id+1,
#         "unstracture": extracted_text
#         }
#     added = await db["user_profiles"].insert_one(format_data)
#     return RedirectResponse("/upload", status_code=303)

# @app.get("/profiles")
# async def get_profiles(request: Request,user_db=Depends(get_authenticated_agent_db)):
#     user, db = user_db
#     # all_profiles = await db["user_profiles"].find({}).to_list(length=None)
#     # return templates.TemplateResponse("profiles.html", {"request": request, "user": user, 'profiles': all_profiles})
#     return templates.TemplateResponse("profiles.html", {"request": request, "user": user})




# @app.post("/api/full-details")
# async def full_details(request: Request, profile_id: int = Form(...), user_db = Depends(get_authenticated_agent_db)):
#     user, db = user_db
#     profile = await db["user_profiles"].find_one({"profile_id": profile_id})
#     return templates.TemplateResponse("full_profile.html", {
#         "request": request,
#         "profile": profile
#     })


# only login user access
# @app.get("/dashboard")
# def dashboard(user: dict = Depends(get_current_user)):
#     return {"message": "Welcome to user dashboard", "user": user}

@app.get("/admin/dashboard")
def admin_dashboard(admin: dict = Depends(admin_required)):
    return {"message": "Welcome to admin dashboard", "admin": admin} 