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
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.tools import tool
from datetime import datetime
import re

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@tool
def calculate_age_from_dob(dob_str: str) -> int:
    """
    Calculate age from various formats of date of birth strings.
    Supports full dates, month-year, year-only, and natural language like '15th Aug 1990'.
    """
    dob_str = dob_str.strip()

    # Remove ordinal suffixes like 1st, 2nd, 3rd, 4th, etc.
    dob_str = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', dob_str, flags=re.IGNORECASE)

    # Handle year-only input
    if re.fullmatch(r"\d{4}", dob_str):
        dob = datetime(int(dob_str), 1, 1)  # Assume Jan 1st of that year
    else:
        date_formats = [
            "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%d %b %Y", "%d %B %Y",           # 15 Aug 1990, 15 August 1990
            "%b %Y", "%B %Y",                 # Aug 1990, August 1990
            "%Y %B", "%Y %b",                 # 1990 August, 1990 Aug
            "%B-%Y", "%b-%Y",                 # August-1990, Aug-1990
        ]

        dob = None
        for fmt in date_formats:
            try:
                dob = datetime.strptime(dob_str, fmt)
                break
            except ValueError:
                continue

        if dob is None:
            raise ValueError(f"Date format not recognized: {dob_str}")

    # Calculate age
    today = datetime.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age


 

async def create_chunks(mongodb_uri, db_name, collection_name):
    # Connect to MongoDB
    client = AsyncIOMotorClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Fetch data from MongoDB
    data = await collection.find({}).to_list(length=None)  # Exclude MongoDB ID
    df = pd.DataFrame(data)

    # Ensure required fields exist
    required_fields = ["pref_age_range", "pref_marital_status", "pref_complexion", "pref_education", "pref_height", 
                    "pref_native_place", "pref_maslak_sect", "pref_no_of_siblings", "pref_work_job", "pref_go_to_dargah", "pref_mother_tongue", "pref_deendari","profile_id","sect", "religious_practice", "full_name", "date_of_birth", "age", "marital_status", 
                "religion", "education", "father" ,"mother", "father_name", "height", "native_place",'occupation','preferences',"religious_sect"]
    for field in required_fields:
        if field not in df.columns:
            df[field] = "unknown"

    # df["text"] = (
    #     df["profile_id"].astype(str) + " \n" +
    #     df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
    #     "Religion: "+ " " + df["religion"].astype(str) + " \n" +
    #     "Education: "+ " " + df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + df["residence"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + df["maslak_sect"].astype(str) + " \n" +
    #     "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
    #     "Preference: "+ " " + df["preferences"].astype(str)
    # )
    
    # pk
    df["text"] = (
        "Name"+" " + df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + df["age"].astype(str) + " \n" +
        "Height: "+ " " + df["height"].astype(str) + " \n" +
        "Education: "+ " " + df["education"].astype(str) + " \n" +
        "sect: "+ " " + df["sect"].astype(str) + " \n" +
        "Preferred Age Range: "+ " " + df["pref_age_range"].astype(str) + " \n" +
        "Preferred Marital Status: "+ " " + df["pref_marital_status"].astype(str) + " \n" +
        "Preferred Complexion: "+ " " + df["pref_complexion"].astype(str) + " \n" +
        "Preferred Education: "+ " " + df["pref_education"].astype(str) + " \n" +
        "Preferred Height: "+ " " + df["pref_height"].astype(str) + " \n" +
        "Preferred Native_place: "+ " " + df["pref_native_place"].astype(str) + " \n" +
        "Preferred Maslak_sect: "+ " " + df["pref_maslak_sect"].astype(str) + " \n" +
        "Preferred Siblings: "+ " " + df["pref_no_of_siblings"].astype(str) + " \n" +
        "Preferred Occupation: "+ " " + df["pref_work_job"].astype(str) + " \n" +
        "Go to dargah: "+ " " + df["pref_go_to_dargah"].astype(str) + " \n" +
        "Preference Mother tongue: "+ " " + df["pref_mother_tongue"].astype(str) + " \n" +
        "Deendar: "+ " " + df["pref_deendari"].astype(str) + " \n" +
        "Preferred location: "+ " " + df["pref_location"].astype(str) + " \n" +
        "Native place: "+ " " + df["native_place"].astype(str) + " \n" +
        "religious practice: "+ " " + df["religious_practice"].astype(str) + " \n" +
        "Preferred own house: "+ " " + df["pref_own_house"].astype(str) + " \n" +
        "Preferences: "+ " " + df["preferences"].astype(str)
    )
    
    df["bio"] = (
        df["profile_id"].astype(str) + " \n" +
        df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
        "Education: "+ " " + df["education"].astype(str) + " \n" +
        "Height: "+ " " + df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + df["residence"].astype(str) + " \n" +
        "Father: "+ " " + df["father"].astype(str) + " \n" +
        "Mother: "+ " " + df["mother"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + df["sect"].astype(str) + " \n" +
        "religious_practice: "+ " " + df["religious_practice"].astype(str) + " \n" +
        "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + df["preferences"].astype(str) + " \n" +
        "Preferred Age Range: "+ " " + df["pref_age_range"].astype(str) + " \n" +
        "Preferred Marital Status: "+ " " + df["pref_marital_status"].astype(str) + " \n" +
        "Preferred Maslak_sect: "+ " " + df["pref_maslak_sect"].astype(str) + " \n" +
        "Preferred location: "+ " " + df["pref_location"].astype(str)
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
        index_path = r"C:\Users\ThinkPad\Desktop\python projects\matrimony_backend\newvectorstore\male_index.faiss"  # Female users search in the male index
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
    distance, faiss_indices = index.search(query_embedding, k=top_k)  # Retrieve extra for filtering
    print(f"FAISS Retrieved Indices: {faiss_indices} and distances: {distance}")
    
    matched_profiles = matched_df.iloc[faiss_indices[0]]  # Ensure only opposite gender profiles are retrieved

    matched_profiles = matched_profiles.head(top_k)  # Return top-k results
    print(f" from search function: {matched_profiles}")
    if matched_profiles.empty:
        return pd.DataFrame(), "❌ No matches found."
    
    return matched_profiles, query_text


def semantic_search_llm(matched_profiles, query_text):
    
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    # gemini-2.0-flash
    # gemini-1.5-flash
    # Generate Explanation Using Gemini
    # Meer sir Prompt
    # combined_input = (
    #     "You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score."

    #         "Matchmaking priorities and weights:"

    #         "Most importantly look at the contents and context under preferences section and try matching *strictly* with that of opposite profile and vice a versa."
    #         "User Preferences with Cross Verification - 50%: *age*, *marital history*, *occupation* , *education*,*family background*, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score."
    #         "Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%."
    #         "Personality & Lifestyle - 10%: Shared interests refine compatibility."
    #         "Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated."
    #         "Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility."

    #         "Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles."

    #         f"User Profile: {query_text}\n\n"
    #         "These are the potential matches:\n"
    #         + "\n\n".join(matched_profiles["bio"].tolist()) + "\n\n"
    #         "Objective Your goal is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process. The Preferences section is given the highest weight to reflect user expectations accurately."
    #     )
    # combined_input = (
    #     f"""You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score.

    #     Matchmaking priorities and weights:

    #     Most importantly look at the contents and context under preferences section and try matching *strictly* with that of opposite profile and vice a versa.
    #     User Preferences with Cross Verification - 50%: *age*, *marital history*, *occupation* , *education*,*family background*, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score.
    #     Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%.
    #     Personality & Lifestyle - 10%: Shared interests refine compatibility.
    #     Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated.
    #     Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility.

    #     Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles.

    #         User Profile: 
    #         {query_text}
    #         "These are the potential matches:
    #         {chr(10).join(matched_profiles["bio"].tolist())}
    #         Objective:
    #          Your goal is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process. The Preferences section is given the highest weight to reflect user expectations accurately

             
    #         IMPORTANT: Output ONLY ONE final match. Do NOT give multiple match options or continue across multiple outputs.
    #         Structure your response exactly in the following format:
            
    #         Match: <Matched Name and Profile ID>
    #         Match Score: <Final Score %>
    #         Score Breakdown:
    #         - User Preferences Match: <value>%
    #         - Religious Alignment: <value>%
    #         - Personality & Lifestyle: <value>%
    #         - Age Difference: <value>%

    #         Reasoning:
    #          <Explain why this profile was selected based on key preferences and religious alignment.>

    #         """
            
    #     )
    

    combined_input = (
        f"""Your role: You are an Expert matchmaking assistant specialised in Muslim marriages.

         Your objective: To provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process. You will evaluate potential matches against a user profile and provide a deterministic match score based on the detailed rules below.
        
        
          **I. Profile Data Key Definitions and Considerations:**

            1.  **Interchangeable Religious Groupings:** The keys 'Religious_Sect', 'Maslak_Sect', 'Religious_Caste', and 'Sect' are to be treated as referring to the same social/religious group. A match occurs if the value in the potential match's corresponding field aligns with the user's specified religious group.
            2.  **Handling Missing Data:** Keys with values "nan" or "None" in either the user profile or the potential match profile must be excluded from the matching and scoring process for that specific criterion. The overall score should be normalized as if that criterion was not applicable.
            3.  **Data Type Specific Matching:**
                * **Numerical (Age, Height):** Matching will be based on ranges or proximity.
                * **Categorical (Education, Sect, Profession, Location):** Matching will be based on equivalence or defined similarity.
            4.  **Name Exclusion:** The 'Name' field must not be considered in the matching or scoring process.
            5.  **Preference Mapping:**
                To ensure meaningful compatibility, preferences should be evaluated from both the user's and the potential match's perspectives
                * Age Compatibility:
                    User's Preferred Age Range ↔ Potential match's actual 'age'
                    Potential match’s Preferred Age Range ↔ User’s actual 'age'
                * Education Preference:
                    User's 'Preferred Education' ↔ Potential match's 'education'
                    Potential match’s 'Preferred Education' ↔ User’s 'education'
                * Height Preference:
                    User's Preferred Height ↔ Potential match's height
                    Potential match’s Preferred Height ↔ User’s height
                * Location & Native Place Alignment:
                    User's Native Place ↔ Potential match's Native Place and Current Location
                    Potential match's Native Place ↔ User's Native Place and Current Location
                * Religious Alignment:
                User's Religious Practices ↔ Potential match's Sect,Maslak_Sect,Religious_Caste, or Religious_Sect
                Potential match's Religious Practices ↔ User's Sect,Maslak_Sect,Religious_Caste, or Religious_Sect.
            6.  **User Preferences Primacy:** The 'preferences' section/keys *within the user's profile* are the primary drivers for scoring against the potential match's actual attributes. Do *not* consider any 'preferences' section that might exist within a *potential match's profile* for scoring purposes.

            **II. Scoring Algorithm (Total Score: 100 points, then expressed as %)**

            The final score is a sum of scores from five main categories. Matches scoring below 50 points (50%) should be explicitly stated as "Not a recommended match" or filtered out.

            **A. User Preferences Match (Max Score: [22] points)**
               *This section reflects how well the potential match meets the user's explicitly stated preferences.*

               1.  **Preferred Age Range Match (Max: [5] points):**
                    * If potential match 'age' is within user 'Preferred Age Range': [5] points.
                    * If potential match 'age' is outside by 1-2 years: [4] points.
                    * If potential match 'age' is outside by 3-5 years: [3] points.
                    * Otherwise: 0 points.
               2.  **Preferred Education Match (Max: [5] points):**
                    * If potential match 'education' exactly matches user 'Preferred Education': [5] points.
                    * If potential match 'education' is one level higher than user 'Preferred Education' (e.g., Masters vs. Bachelors): [5] points.
                    * If potential match 'education' is one level lower: [4] points.
                    * Otherwise: 0 points. (user profile education vs potential profile education if it is not match.)
               3.  **Preferred Height Match (Max: [5] points):**
                    * If potential match 'height' is within user 'Preferred Height' range (height will in this format e.g., 5'6" - 5'10",5ft etc): [5] points.
                    * If potential match 'height' exactly matches user 'Preferred Height' range: [5] points.
                    * If potential match 'height' is outside by 1-2 inches: [3] points.
                    * Otherwise: 0 points.
               4.  **Preferred Native Place Match (vs. Location/Native Place) (Max: [2] points):**
                    * If user's 'Native Place' exactly matches potential match's 'native place': [2] points.
                    * If user's 'Native Place' exactly matches potential match's 'location' (but not 'native place'): 2 points.
                    * If user's 'Native Place' (City/Town) is in the same District/State as potential match's 'native place' or 'location': 1 points.
                    * Otherwise: 0 points.
               5.  **User's Religious Practices Match (vs. Sect/Maslak_Sect/Preferences) (Max: [5] points):**
                    * If user's 'Religious Practices' (e.g., specific Maslak or level of observance if detailed) exactly matches potential match's 'Sect'/'Maslak_Sect'/'Preferences': [5] points.
                    * If considered compatible but not an exact match (e.g., different but mutually acceptable Maslaks - YOU define this): [2] points.
                    * Otherwise: 0 points.

            **B. Optential Preferences Match (Max Score: [22] points)**
               *This section reflects how well the user match meets the optential's explicitly stated preferences.*

               1.  **Preferred Age Range Match (Max: [5] points):**
                    **Note: Calculate date of birth from current date.**
                    * If user match 'age' is within optential 'Preferred Age Range': [5] points.
                    * If user match 'age' is outside by 1-2 years: [4] points.
                    * If user match 'age' is outside by 3-5 years: [3] points.
                    * Otherwise: 0 points.
               2.  **Preferred Education Match (Max: [5] points):**
                    * If user match 'education' exactly matches potential 'Preferred Education': [5] points.
                    * If user match 'education' is one level higher than optential 'Preferred Education' (e.g., Masters vs. Bachelors): [5] points.
                    * If user match 'education' is one level lower: [4] points.
                    * Otherwise: 0 points. (optential profile education vs user profile education if it is not match.)
               3.  **Preferred Height Match (Max: [5] points):**
                    * If user match 'height' is within optential 'Preferred Height' range (height will in this format e.g., 5'6" - 5'10",5ft etc): [5] points.
                    * If user match 'height' exactly matches optential 'Preferred Height' range: [5] points.
                    * If user match 'height' is outside by 1-2 inches: [3] points.
                    * Otherwise: 0 points.
               4.  **Preferred Native Place Match (vs. Location/Native Place) (Max: [2] points):**
                    * If optential's 'Native Place'/'/Location' exactly matches user match's 'native place'/'Location': [2] points.
                    * If optential's 'Native Place' exactly matches user match's 'location' (but not 'native place'): 2 points.
                    * If optential's 'Native Place' (City/Town) is in the same District/State as user match's 'native place' or 'location': 1 points.
                    * Otherwise: 0 points.
               5.  **optential's Religious Practices Match (vs. Sect/Maslak_Sect/Preferences) (Max: [5] points):**
                    * If optential's 'Religious Practices' (e.g., specific Maslak or level of observance if detailed) exactly matches user match's 'Sect'/'Maslak_Sect'/'Preferences': [5] points.
                    * If considered compatible but not an exact match (e.g., different but mutually acceptable Maslaks - Deobandi vs Ahl-e-Hadith or Ahle Hadees (Salafi),Ahl-e-Hadith or Ahle Hadees (Salafi) vs Tablighi Jamaat and Tablighi Jamaat vs Deobandi): [2] points.
                    * Otherwise: 0 points.

            **C. Religious Alignment (Max Score: [20] points)**
               *This section assesses broader religious compatibility beyond direct user preferences if not fully covered above.*
                *Consider 'Namazi','Deendar','Deen' all as one and the Same*

               1.  **Sect/Maslak/Preferences Match (if not covered with higher weight in User Preferences) (Max: [10] points):**
                    * Exact match between user's 'Sect' (or equivalent) and potential match's 'Sect' (or equivalent): [10] points.
                    * Compatible (as defined by you, e.g., within Ahlus Sunnah wal Jama'ah if user is Sunni): [5] points.
                    * Mismatch (e.g., Shia vs Sunni, if not acceptable and user is not mention): 0 points.
               2.  **Religious Observance Level (if available as a field, e.g., 'Practicing', 'Moderately Practicing', 'Not Practicing') (Max: [10] points):**
                    * Exact Match: [5] points.
                    * One level difference (e.g., User 'Practicing', Match 'Moderately Practicing'): [5] points.
                    * Otherwise: 0 points.

            **D. Personality & Lifestyle (Max Score: [20] points)**
               *This section requires defining specific fields that contribute to personality and lifestyle compatibility.*

              1.  **Profession Compatibility (Max: [20] points):**
                    * If user has a 'Preferred Profession Type' (e.g., "Doctor", "Engineer", "Teacher", "Business Owner", "Homemaker", "Teacher") and also potential match's 'Preferred Profession' fits: [20] points.
                    * If no 'Preferred Profession Type' by user, but potential match's profession is generally considered stable/respected (e.g., "Doctor", "Engineer", "Teacher", "Business Owner", "Homemaker", "Teacher"): [4] points.
                    * Otherwise: 0 points.

            **E. Age Difference (Max Score: [15] points)**
                *This score is independent of the 'Preferred Age Range' and reflects general societal/cultural compatibility in age. This can be a nuanced score.*
                *Note:- if age is not mention/not available in user/potential profiles use dob from profiles for age comparing age range. calculate dob from current date.*
                * If potential match's 'age' is within user's 'Preferred Age Range': [15] points (full points for this separate category if they meet the preference).
                *If Age difference 0-3 years: [8] points.
                *If Age difference 4-7 years: [5] points.
                * Otherwise: 0 points.
                * If user has NO 'Preferred Age Range' specified, or to refine further:
                    * Age difference 0-3 years (male older or same age): [5] points.
                    * Age difference 4-7 years (male older): [5] points.
                    * Age difference 0-1 year (female older): [5] points.
                    * Other cases: [A_Other] points or 0.
                    *(Define these sub-points carefully based on your target users' expectations. Ensure [A1/A2/A3/A_Other] sum up to X4 or that X4 is awarded if the preferred range is met).*

            **III. Normalization and Final Score Calculation:**

            1.  **Calculate Score for Each Category (A, B, C, D, E):** Sum the points awarded for individual criteria within that category.
            2.  **Handle Missing Data for Normalization:** If a criterion was skipped due to "nan" or "None" in *essential* fields for that criterion, the Max Score for that category should be adjusted downwards by the Max points of the skipped criterion. For example, if 'Preferred Height' was "nan", then the Max Score for 'User Preferences Match' becomes (X1 - P3). The achieved score is then (Achieved_User_Preference_Points / (X1 - P3)) * X1 to keep it scaled to X1 for the final sum, OR you calculate the percentage achieved within the *available* criteria for that section. *The latter is usually simpler: Score for Category A = (Sum of points for criteria met in A) / (Sum of max points for criteria in A for which data was available) * X1.*
            3.  **Total Score = Score A + Score B + Score C + Score D + Score E.** This total will be out of (X1+X2+X3+X4+X5) points. Ensure X1+X2+X3+X4+X5 = 100 if you want a direct percentage.
            4.  **Final Match Score % = Total Score.

            **IV. Response Structure:**

            Structure your response *exactly* in the following format for each recommended match (score >= 50%):

            Match: <Matched Name> And Profile Id: <Profile ID %>%
            Match Score: <Final Score %>%
            Score Breakdown:
            - User Preferences Match: <(Achieved_A / Max_Possible_A_With_Available_Data) * 100 %>%
            - Optential Preferences Match: <(Achieved_B / Max_Possible_B_With_Available_Data) * 100 %>%
            - Religious Alignment: <(Achieved_C / Max_Possible_C_With_Available_Data) * 100 %>%
            - Personality & Lifestyle: <(Achieved_D / Max_Possible_D_With_Available_Data) * 100 %>%
            - Age Difference: <(Achieved_E / Max_Possible_E_With_Available_Data) * 100 %>%

            ---

            **Make sure X1 + X2 + X3 + X4 + X5 sum to your desired total (e.g., 100 points for easy percentage conversion).**
            You need to list *all* relevant fields from your profiles that would be used in these calculations.
            For "Personality & Lifestyle," if you have generic "preference" text fields, the LLM can *try* to interpret them, but it will be less deterministic than matching on specific categorical fields. For highest determinism, rely on structured data.

                    Following is the user profile (in key:value format) for which the match has to be found:
                        {query_text}
                    Following are the potential matches for the above profile:
                    {chr(10).join(matched_profiles["bio"].tolist())}
                    """
            
        )
    # combined_input = (
    #     f"""You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score.

    #     Matchmaking priorities and weights:

    #     Most importantly look at the contents and context under preferences section and try matching *strictly* with that of opposite profile and vice a versa.
    #     User Preferences with Cross Verification - 50%: *age*, *marital history*, *occupation* , *education*,*family background*, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score.
    #     Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%.
    #     Personality & Lifestyle - 10%: Shared interests refine compatibility.
    #     Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated.
    #     Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility.

    #     Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles.
    #         IMPORTANT: Output ALL matched profiles whose final score is greater than 50%.
    #             Structure your response exactly in the following format:
                
    #             Match: <Matched Name and Profile ID>
    #             Match Score: <Final Score %>
    #             Score Breakdown:
    #             - User Preferences Match: <value>%
    #             - Religious Alignment: <value>%
    #             - Personality & Lifestyle: <value>%
    #             - Age Difference: <value>%

    #             Reasoning:
    #             <Explain why this profile was selected based on key preferences and religious alignment.>

    #             If no suitable match is found (score below 50%), state clearly: No match found due to insufficient compatibility.
    #         User Profile: 
    #         {query_text}
    #         "These are the potential matches:
    #         {chr(10).join(matched_profiles["bio"].tolist())}
    #         Objective:
    #          Your goal is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process. The Preferences section is given the highest weight to reflect user expectations accurately

             

    #         """
            
    #     )
    # Send to Gemini LLM
    messages = [
        SystemMessage(content="You are an AI assistant that helps match profiles for a matrimonial platform."),
        HumanMessage(content=combined_input)
    ]
    # pdb.set_trace()

    result = gemini_model.invoke(messages)
    
    return result.content
    # tools = [calculate_age_from_dob]
    # llm_with_tools = gemini_model.bind_tools(tools)
    # ai_message = llm_with_tools.invoke(combined_input)
    # # Tool usage
    # messages = [HumanMessage(content=combined_input), ai_message]
    # print(f"human message: {messages}")
    # # pdb.set_trace()
    # for tool_call in ai_message.tool_calls:
    #     selected_tool = {"calculate_age_from_dob": calculate_age_from_dob}[tool_call["name"].lower()]
    #     tool_msg = selected_tool.invoke(tool_call)
    #     messages.append(tool_msg)
    # print(messages)    
    # final_result = llm_with_tools.invoke(messages)
    # return final_result.content



def transform_llm_response(llm_response):
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Define the user profile schema
    class UserProfile(BaseModel):
        name: str
        age_range: str
        marital_status: str
        religion: str
        location: str
        education: str
        preferences: str

    # Define a model for Match Evaluation scores
    class MatchScore(BaseModel):
        user_preferences: int
        religious_alignment: int
        personality_lifestyle: int
        age: int
        total_score: int
        compatibility: str
        Reasoning: str

    # Define a model for each match
    class Match(BaseModel):
        profile_id: int = Field(description="Exctract profile_id")
        name: str
        age: Optional[str] = Field(description="Write only the age or date_of_birth if available", default="Unknown")
        marital_status: str 
        occupation: str
        education: str
        family_background: Optional[str] = "Unknown"
        native_place: str
        maslak_sect: Optional[str] = Field(description="Write only the maslak or sect if available", default="Unknown")
        religious_alignment: Optional[str] = "Unknown"
        personality_lifestyle: Optional[str] = "Unknown"
        preferences: str
        score_breakdown: MatchScore

    # Define a model for the overall match analysis
    class MatchAnalysis(BaseModel):
        user_profile: UserProfile
        matches: List[Match]
        conclusion: str
    
    structured_model = gemini_model.with_structured_output(MatchAnalysis)
    
    result = structured_model.invoke(llm_response)
    result_dict = dict(result)
    
    return result_dict
    
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
    required_fields = ["pref_age_range", "pref_marital_status", "pref_complexion", "pref_education", "pref_height", 
                    "pref_native_place", "pref_maslak_sect", "pref_no_of_siblings", "pref_work_job", "pref_go_to_dargah", "pref_mother_tongue", "pref_deendari","profile_id","sect", "full_name", "date_of_birth", "age", "marital_status", 
                "religion","location", "education","mother","father","maslak_sect", "height","religious_practice" ,"native_place",'occupation','preferences',"go_to_dargah"]
    for field in required_fields:
        if field not in male_df.columns:
            male_df[field] = "unknown"
    for field in required_fields:
        if field not in female_df.columns:
            female_df[field] = "unknown"
            
    # male_df["text"] = (
    #     male_df["profile_id"].astype(str) + " \n" +
    #     male_df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + male_df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + male_df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + male_df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + male_df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + male_df["complexion"].astype(str) + " \n" +
    #     "Religion: "+ " " + male_df["religion"].astype(str) + " \n" +
    #     "Education: "+ " " + male_df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + male_df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + male_df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + male_df["residence"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + male_df["maslak_sect"].astype(str) + " \n" +
    #     "occupation: "+ " " + male_df["occupation"].astype(str) + " \n" +
    #     "Preference: "+ " " + male_df["preferences"].astype(str)
    # )
    # female_df["text"] = (
    #     female_df["profile_id"].astype(str) + " \n" +
    #     female_df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + female_df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + female_df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + female_df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + female_df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + female_df["complexion"].astype(str) + " \n" +
    #     "Religion: "+ " " + female_df["religion"].astype(str) + " \n" +
    #     "Education: "+ " " + female_df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + female_df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + female_df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + female_df["residence"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + female_df["maslak_sect"].astype(str) + " \n" +
    #     "occupation: "+ " " + female_df["occupation"].astype(str) + " \n" +
    #     "Preference: "+ " " + female_df["preferences"].astype(str)
    # )
    # normal keys
    male_df["text"] = (
        male_df["profile_id"].astype(str) + " \n" +
        male_df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + male_df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + male_df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + male_df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + male_df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + male_df["complexion"].astype(str) + " \n" +
        "Education: "+ " " + male_df["education"].astype(str) + " \n" +
        "Height: "+ " " + male_df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + male_df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + male_df["residence"].astype(str) + " \n" +
        "Location: "+ " " + male_df["location"].astype(str) + " \n" +
        "Father: "+ " " + male_df["father"].astype(str) + " \n" +
        "Mother: "+ " " + male_df["mother"].astype(str) + " \n" +
        "sect: "+ " " + male_df["sect"].astype(str) + " \n" +
        "religious_practice: "+ " " + male_df["religious_practice"].astype(str) + " \n" +
        "go_to_dargah: "+ " " + male_df["go_to_dargah"].astype(str) + " \n" +
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
        "Education: "+ " " + female_df["education"].astype(str) + " \n" +
        "Height: "+ " " + female_df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + female_df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + female_df["residence"].astype(str) + " \n" +
        "Location: "+ " " + female_df["location"].astype(str) + " \n" +
        "Father: "+ " " + female_df["father"].astype(str) + " \n" +
        "Mother: "+ " " + female_df["mother"].astype(str) + " \n" +
        "sect: "+ " " + female_df["sect"].astype(str) + " \n" +
        "religious_practice,: "+ " " + female_df["religious_practice"].astype(str) + " \n" +
        "go_to_dargah: "+ " " + female_df["go_to_dargah"].astype(str) + " \n" +
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


# from datetime import datetime

# def calculate_age_from_dob(dob_str: str) -> int:
#     """
#     Calculate age from date of birth string (format: YYYY-MM-DD).
#     """
#     dob = datetime.strptime(dob_str, "%Y-%m-%d")
#     today = datetime.today()
#     age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
#     return age

# age = calculate_age_from_dob("2002-8-24")
# print(age)