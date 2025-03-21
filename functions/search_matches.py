import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_core.prompts import ChatPromptTemplate
import pdb

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

async def create_chunks(mongodb_uri, db_name, collection_name):
    # Connect to MongoDB
    client = AsyncIOMotorClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Fetch data from MongoDB
    data = await collection.find({}).to_list(length=None)  # Exclude MongoDB ID
    df = pd.DataFrame(data)

    # Ensure required fields exist
    required_fields = ["profile_id", "full_name", "date_of_birth", "age", "marital_status", 
                       "gender", "religion", "education", "height", "native_place", "preferences"]
    for field in required_fields:
        if field not in df.columns:
            df[field] = "unknown"

    df["text"] = (
        df["profile_id"].astype(str) + " \n" +
        df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
        "Religion: "+ " " + df["religion"].astype(str) + " \n" +
        "Education: "+ " " + df["education"].astype(str) + " \n" +
        "Height: "+ " " + df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + df["residence"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + df["maslak_sect"].astype(str) + " \n" +
        "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + df["preferences"].astype(str)
    )

    # Convert the combined text to a list
    texts = df["text"].tolist()
    return texts,df

# Function to normalize embeddings
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def extract_indices_from_vector(df, user_name,top_k):
    
    # Get user profile
    user_profile = df[df["full_name"] == user_name]
    if user_profile.empty:
        return pd.DataFrame(), "❌ User not found."

    user_gender = user_profile.iloc[0]["gender"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Select the appropriate FAISS index
    if user_gender == "Male":
        matched_df = df[df["gender"] == "Female"]  # Male searches for females
        index_path = r"C:\Users\ThinkPad\Desktop\python projects\matrimony_backend\newvectorstore\female_index.faiss"  # Male users search in the female index
        opposite_gender = "Female"
    elif user_gender== "Female":
        matched_df = df[df["gender"] == "Male"]  # Female searches for males
        index_path = r"C:\Users\ThinkPad\Desktop\python projects\matrimony_backend\newvectorstore\female_index.faiss"  # Female users search in the male index
        opposite_gender = "Male"
    else:
        return pd.DataFrame(), "❌ Invalid gender."

    print(f"🔍 User Gender: {user_gender}, Searching in: {index_path} (Looking for {opposite_gender})")

    # Load the FAISS index
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        return pd.DataFrame(), f"❌ Error loading FAISS index: {str(e)}"

    # Encode user profile text
    query_text = user_profile.iloc[0]["text"]
    query_embedding = model.encode([query_text]).astype("float32")
    query_embedding = normalize_embeddings(query_embedding)

    # Search FAISS
    _, faiss_indices = index.search(query_embedding, k=top_k)  # Retrieve extra for filtering
    print(f"FAISS Retrieved Indices: {faiss_indices}")
    
    matched_profiles = matched_df.iloc[faiss_indices[0]]  # Ensure only opposite gender profiles are retrieved

    matched_profiles = matched_profiles.head(top_k)  # Return top-k results

    if matched_profiles.empty:
        return pd.DataFrame(), "❌ No matches found."
    
    return matched_profiles, query_text

def semantic_search_llm(matched_profiles, query_text):
    
    gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    # Generate Explanation Using Gemini
    # combined_input = (
    #     "You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score.\n\n"

    #     "Matchmaking priorities and weights:\n"
    #     "1. User Preferences with Cross Verification - 50%: Age, marital history, occupation, education, family background, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score.\n"
    #     "2. Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%.\n"
    #     "3. Personality & Lifestyle - 10%: Shared interests refine compatibility.\n"
    #     "4. Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated.\n\n"

    #     "Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility.\n\n"

    #     "Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles.\n\n"

    #     f"User Profile: {query_text}\n\n"
    #     "These are the potential matches:\n"
    #     + "\n\n".join(matched_profiles["text"].tolist()) + "\n\n"
    #     "Your objective is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process."
    # )
    combined_input = (
        "You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score."

            "Matchmaking priorities and weights:"

            "Most importantly look at the contents and context under preferences section and try matching *strictly* with that of opposite profile and vice a versa."
            "User Preferences with Cross Verification - 50%: *age*, *marital history*, *occupation* , *education*,*family background*, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score."
            "Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%."
            "Personality & Lifestyle - 10%: Shared interests refine compatibility."
            "Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated."
            "Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility."

            "Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles."

            f"User Profile: {query_text}\n\n"
            "These are the potential matches:\n"
            + "\n\n".join(matched_profiles["text"].tolist()) + "\n\n"
            "Objective Your goal is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process. The Preferences section is given the highest weight to reflect user expectations accurately."
        )
    
    # Send to Gemini LLM
    messages = [
        SystemMessage(content="You are an AI assistant that helps match profiles for a matrimonial platform."),
        HumanMessage(content=combined_input)
    ]

    result = gemini_model.invoke(messages)
    
    return result.content


def transform_llm_response(llm_response):
    gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    messages = [
    ("system", "You are an AI assisstence"),
    ("human", """From the following text, prepare a dictionary that contains following keys:
     Profile, Age, Marital History, Occupation. Prepare nested dictionaries under Age, Marital History,
      and Occupation keys and have two more keys namely score and remarks. In remarks keep everything apart from score and age. 
     The text is as follows: {text}""")]
    
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt = prompt_template.invoke({"text": llm_response})
    result = gemini_model.invoke(prompt)
    return result.content
    
async def create_faiss_index(mongodb_uri, db_name, collection_name):
    """Create separate FAISS indexes for male and female profiles."""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Normalize gender labels before FAISS indexing
    # df["gender"] = df["gender"].str.strip().str.lower()

    # gender_mapping = {
    #     "female": "Female",
    #     "male": "Male",
    #     "f": "Female",
    #     "m": "Male",
    #     "only for ladies": "Female",
    #     "nil": "Unknown"
    # }
    
    # df["gender"] = df["gender"].replace(gender_mapping)
    # df = df[df["gender"].isin(["Male", "Female"])]  # Keep only valid profiles
    # Connect to MongoDB
    client = AsyncIOMotorClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Fetch data from MongoDB
    male_data = await collection.find({"gender": "Male"}).to_list(length=None)  # Exclude MongoDB ID
    female_data = await collection.find({"gender": "Female"}).to_list(length=None)  # Exclude MongoDB ID
    male_df = pd.DataFrame(male_data)
    female_df = pd.DataFrame(female_data)
    
    # Ensure required fields exist
    required_fields = ["profile_id", "full_name", "date_of_birth", "age", "marital_status", 
                       "gender", "religion", "education", "height", "native_place", "preferences"]
    for field in required_fields:
        if field not in male_df.columns:
            male_df[field] = "unknown"
    for field in required_fields:
        if field not in female_df.columns:
            female_df[field] = "unknown"
            
    male_df["text"] = (
        male_df["profile_id"].astype(str) + " \n" +
        male_df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + male_df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + male_df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + male_df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + male_df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + male_df["complexion"].astype(str) + " \n" +
        "Religion: "+ " " + male_df["religion"].astype(str) + " \n" +
        "Education: "+ " " + male_df["education"].astype(str) + " \n" +
        "Height: "+ " " + male_df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + male_df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + male_df["residence"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + male_df["maslak_sect"].astype(str) + " \n" +
        "occupation: "+ " " + male_df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + male_df["preferences"].astype(str)
    )
    female_df["text"] = (
        female_df["profile_id"].astype(str) + " \n" +
        female_df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + female_df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + female_df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + female_df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + female_df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + female_df["complexion"].astype(str) + " \n" +
        "Religion: "+ " " + female_df["religion"].astype(str) + " \n" +
        "Education: "+ " " + female_df["education"].astype(str) + " \n" +
        "Height: "+ " " + female_df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + female_df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + female_df["residence"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + female_df["maslak_sect"].astype(str) + " \n" +
        "occupation: "+ " " + female_df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + female_df["preferences"].astype(str)
    )
    texts = female_df["text"].tolist()
    
    male_embeddings = model.encode(male_df["text"].tolist()).astype("float32")
    female_embeddings = model.encode(female_df["text"].tolist()).astype("float32")
    
    #     # Normalize embeddings
    male_embeddings /= np.linalg.norm(male_embeddings, axis=1, keepdims=True)
    female_embeddings /= np.linalg.norm(female_embeddings, axis=1, keepdims=True)

    # Create FAISS indexes **SEPARATELY**
    male_index = faiss.IndexFlatL2(male_embeddings.shape[1])
    female_index = faiss.IndexFlatL2(female_embeddings.shape[1])

    male_index.add(male_embeddings)  # type: ignore # Male Index (to match Females)
    female_index.add(female_embeddings)  # type: ignore # Female Index (to match Males)

    os.makedirs("newvectorstore", exist_ok=True)

    # Save FAISS indexes separately
    faiss.write_index(male_index, "newvectorstore/male_index.faiss")   # Male users will search here (matches Female)
    faiss.write_index(female_index, "newvectorstore/female_index.faiss")  # Female users will search here (matches Male)

    print("✅ FAISS indexes created successfully.")
