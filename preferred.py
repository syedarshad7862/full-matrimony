# from pymongo import MongoClient, UpdateOne
# import os
# import re
# from datetime import datetime
# Mongo_uri = os.getenv("MONGO_URI")

# # connect Mongo
# client = MongoClient(Mongo_uri)
# db = client['matrimony_adnan_67cc880d12d32259d0b13a05']
# collection = db["user_profiles"]

# # index = collection.create_index({"education": 'text'},default_language='none')
# # profiles = collection.find({"profile_id": 271}).to_list(length=None)

# # print(profiles)


# younger_age = 3
# older_age = 1
# min_allowed_age = 18


# # === Helper: Calculate age from DOB
# def calculate_age(dob_str):
#     """
#     Calculate age from various formats of date of birth strings.
#     Supports full dates, month-year, year-only, and natural language like '15th Aug 1990'.
#     """
#     try:
#         if not isinstance(dob_str, str):
#             return None

#         dob_str = dob_str.strip()

#         # Remove ordinal suffixes like 1st, 2nd, 3rd, etc.
#         dob_str = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', dob_str, flags=re.IGNORECASE)

#         # Handle year-only input like "1997"
#         if re.fullmatch(r"\d{4}", dob_str):
#             dob = datetime(int(dob_str), 1, 1)  # Assume Jan 1st
#         else:
#             date_formats = [
#                 "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
#                 "%d %b %Y", "%d %B %Y",             # 15 Aug 1990, 15 August 1990
#                 "%b %Y", "%B %Y",                   # Aug 1990, August 1990
#                 "%Y %B", "%Y %b",                   # 1990 August, 1990 Aug
#                 "%B-%Y", "%b-%Y",                   # August-1990, Aug-1990
#                 "%Y/%B", "%Y/%b",                   # 1990/August, 1990/Aug
#                 "%Y.%m.%d", "%d.%m.%Y"              # 1990.08.15, 15.08.1990
#             ]

#             dob = None
#             for fmt in date_formats:
#                 try:
#                     dob = datetime.strptime(dob_str, fmt)
#                     break
#                 except ValueError:
#                     continue

#             if dob is None:
#                 raise ValueError(f"Unrecognized DOB format: '{dob_str}'")

#         # Calculate age
#         today = datetime.today()
#         age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
#         return age

#     except Exception as e:
#         print(f"Failed to parse DOB '{dob_str}': {e}")
#         return None

# # === Fetch users with age or dob
# users_cursor = collection.find({
#     "$or": [
#         {"age": {"$exists": True}},
#         {"date_of_birth": {"$exists": True}}
#     ]
# })

# bulk_updates = []
# skipped_users = []

# # print(calculate_age("August 1997"))
# for user in users_cursor:
#     user_age = None

#     if "age" in user:
#         user_age = user["age"]
#     elif "date_of_birth" in user:
#         user_age = calculate_age(user["date_of_birth"])
#         if user_age is not None:
#             # Optional: store calculated age in DB
#             collection.update_one(
#                 {"_id": user["_id"]},
#                 {"$set": {"age": user_age}}
#             )

#     if user_age is None:
#         skipped_users.append(str(user["_id"]))
#         continue

#     # === Calculate preferred age range
#     min_age = max(int(user_age) - younger_age, min_allowed_age)
      
#     max_age = int(user_age) + older_age
#     preferred_age_range = f"{min_age}-{max_age}"

#     # === Prepare bulk update
#     update_op = UpdateOne(
#         {"_id": user["_id"]},
#         {"$set": {"pref_age_range": preferred_age_range}}
#     )
#     bulk_updates.append(update_op)

# # === Execute bulk update
# if bulk_updates:
#     result = collection.bulk_write(bulk_updates)
#     print(f"Matched: {result.matched_count}, Modified: {result.modified_count}")
# else:
#     print("No users were eligible for update.")

# # === Report skipped users (missing age and dob)
# if skipped_users:
#     print(f"Skipped {len(skipped_users)} users due to missing age and dob.")
 



# for single profile
# from pymongo import MongoClient
# from bson import ObjectId

# # === Sample Inputs ===
# user_id = "67f0901136dcb1f14017db14"  
# younger_age_offset = 3
# older_age_offset = 0

# # === Connect to MongoDB ===
# client = MongoClient("mongodb://localhost:27017/")
# db = client["matrimony_adnan_67cc880d12d32259d0b13a05"]
# users_collection = db["user_profiles"]

# # === Fetch User Age ===
# user_profile = users_collection.find_one({"_id": ObjectId(user_id)})
# print(user_profile)
# if not user_profile or "age" not in user_profile:
#     print("User not found or age field is missing.")
# else:
#     user_age = user_profile["age"]
    
#     # === Calculate Preferred Age Range ===
#     min_age = max(int(user_age) - younger_age_offset, 18)  # Prevents age below 18
#     max_age = int(user_age) + older_age_offset

#     preferred_age_range = f"{min_age}-{max_age}"
#     print(preferred_age_range)
#     # === Update in MongoDB ===
#     users_collection.update_one(
#         {"_id": ObjectId(user_id)},
#         {"$set": {"pref_age_range": preferred_age_range}}
#     )

#     print(f"Updated preferred age range: {preferred_age_range}")



from pymongo import MongoClient, UpdateOne
import os
import re
from datetime import datetime

Mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(Mongo_uri)
db = client['matrimony_adnan_67cc880d12d32259d0b13a05']
collection = db["user_profiles"]

# Configurable values
younger_age = 3
older_age = 1
min_allowed_age = 18

# === Helper: Calculate age from DOB ===
def calculate_age(dob_str):
    """
    Calculate age from various formats of date of birth strings.
    Supports full dates, month-year, year-only, and natural language like '15th Aug 1990'.
    """
    try:
        if not isinstance(dob_str, str):
            return None

        dob_str = dob_str.strip()
        dob_str = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', dob_str, flags=re.IGNORECASE)

        if re.fullmatch(r"\d{4}", dob_str):
            dob = datetime(int(dob_str), 1, 1)
        else:
            date_formats = [
                "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
                "%d %b %Y", "%d %B %Y",
                "%b %Y", "%B %Y",
                "%Y %B", "%Y %b",
                "%B-%Y", "%b-%Y",
                "%Y/%B", "%Y/%b",
                "%Y.%m.%d", "%d.%m.%Y"
            ]

            dob = None
            for fmt in date_formats:
                try:
                    dob = datetime.strptime(dob_str, fmt)
                    break
                except ValueError:
                    continue

            if dob is None:
                raise ValueError(f"Unrecognized DOB format: '{dob_str}'")

        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age

    except Exception as e:
        print(f"Failed to parse DOB '{dob_str}': {e}")
        return None

# === Fetch users with age or dob ===
users_cursor = collection.find({
    "$or": [
        {"age": {"$exists": True}},
        {"date_of_birth": {"$exists": True}}
    ]
})

bulk_updates = []
skipped_users = []

for user in users_cursor:
    user_age = None

    # Try getting age if it’s a valid number
    if "age" in user and isinstance(user["age"], (int, float)):
        user_age = int(user["age"])

    # If age not usable, calculate from DOB
    elif "date_of_birth" in user:
        user_age = calculate_age(user["date_of_birth"])
        if user_age is not None:
            # Add age to DB as part of bulk update
            bulk_updates.append(
                UpdateOne({"_id": user["_id"]}, {"$set": {"age": user_age}})
            )

    if user_age is None:
        skipped_users.append(str(user["_id"]))
        continue

    # === Calculate preferred age range ===
    min_age = max(int(user_age) - younger_age, min_allowed_age)
    max_age = int(user_age) + older_age
    preferred_age_range = f"{min_age}-{max_age}"

    # === Add preferred age range to bulk update ===
    bulk_updates.append(
        UpdateOne(
            {"_id": user["_id"]},
            {"$set": {"pref_age_range": preferred_age_range}}
        )
    )

# === Execute bulk update ===
if bulk_updates:
    try:
        result = collection.bulk_write(bulk_updates)
        print(f"Matched: {result.matched_count}, Modified: {result.modified_count}")
    except Exception as e:
        print(f"Error during bulk update: {e}")
else:
    print("No users were eligible for update.")

# === Report skipped users ===
if skipped_users:
    print(f"Skipped {len(skipped_users)} users due to missing or invalid age and DOB.")

