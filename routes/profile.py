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
from bson import ObjectId
from bson.errors import InvalidId
from typing import Optional
app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# database
MONGO_URI = os.getenv("MONGO_URI")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix='/profile', tags=['Profile'])

# form route
@router.get("/add-profile")
async def get_profile(request: Request,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db  # Unpack user & database from function
    previous_id = await db["user_profiles"].count_documents({})
        # biodata["profile_id"] = previous_id+1
    print(previous_id)
    return templates.TemplateResponse("add_bio.html", {"request": request, "user": user})


from pymongo.errors import PyMongoError
from fastapi import HTTPException

@router.post("/add-profile")
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
        return RedirectResponse(url="/profile/add-profile", status_code=303)

    except PyMongoError as e:
        print(f"Database Error: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="Database error occurred while adding profile")

    except Exception as e:
        print(f"Unexpected Error: {e}")  # Log the error
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    

@router.get("/edit-profiles/{profile_id}")
async def edit_profile(request: Request,profile_id: str,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    try:
        obj_id = ObjectId(profile_id)
    except InvalidId:
        return {"error": "Invalid ID"}
    profile = await db["user_profiles"].find_one({'_id': obj_id})
    return templates.TemplateResponse("edit_profile.html", {"request": request, "user": user, 'profile':profile})

@router.post("/edit-profile/{profile_id}")
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
    return RedirectResponse(url="/profile/api/filtered-profiles", status_code=303)

# profile completion filter
@router.get("/api/filtered-profiles")
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