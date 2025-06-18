from fastapi import FastAPI, APIRouter, Form,Request,HTTPException,Depends,Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
import datetime
import os
from passlib.context import CryptContext
from config.db import client,agents_db
from functions.search_matches import create_chunks,extract_indices_from_vector,semantic_search_llm,create_faiss_index,transform_llm_response
from utils import get_authenticated_agent_db
from auth import get_current_user,admin_required
from starlette.responses import HTMLResponse
from bson import ObjectId
import csv
import io
import time

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# database
MONGO_URI = os.getenv("MONGO_URI")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix='/match', tags=['Match'])

@router.get("/find")
async def find_matches(request: Request,user_db=Depends(get_authenticated_agent_db)):
    user, db = user_db
    all_profiles = await db["user_profiles"].find({}).to_list(length=None)
    # print(all_profiles)
    return templates.TemplateResponse("find_matches.html", {"request": request, "user": user, "profiles": all_profiles,"selected_profile": None}) 

@router.post("/show_matches", response_class=HTMLResponse)
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
    # print(f"query_profile: {query_text} and potential_profile: {matched_profiles.to_dict()}")
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
    # pdb.set_trace()
    result = transform_llm_response(llm_response)
    # Convert all Match dataclass objects to plain dictionaries
    matches = [m.dict() for m in result['matches']]
    # print(f"selected_profile: {selected_profile} \n\n Matches Profiles: {matches} \n\n matched_profiles: {matched_profiles}")
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

@router.post("/create-vectors")
async def create_vector(request: Request, user_db= Depends(get_authenticated_agent_db)):
    user, db = user_db
    await create_faiss_index(MONGO_URI,db.name,"user_profiles")
    print(MONGO_URI,db.name,"user_profiles")
    return {"message": "Created vectors successfully!"}

@router.post("/download_matches_csv")
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


@router.get("/api/full-details")
async def full_details_json(profile_id: int, user_db = Depends(get_authenticated_agent_db)):
    user, db = user_db
    profile = await db["user_profiles"].find_one({"profile_id": profile_id})
    if not profile:
        return JSONResponse(status_code=404, content={"error": "Profile not found"})
    # Convert ObjectId to str
    profile["_id"] = str(profile["_id"])
    print(profile)
    return profile  # FastAPI will return JSON

