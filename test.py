# from datetime import datetime
# def calculate_age_from_dob(dob_str: str) -> int:
#     """
#     Calculate age from date of birth string (format: YYYY-MM-DD).
#     """
#     dob = datetime.strptime(dob_str, "%d-%m-%Y")
#     today = datetime.today()
#     age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
#     return age

# age = calculate_age_from_dob("22-10-2001")
# print(age)

# num = input("enter the n:")
# print(num*2)

from pymongo import MongoClient, UpdateOne
import os
Mongo_uri = os.getenv("MONGO_URI")

# connect Mongo
client = MongoClient(Mongo_uri)
db = client['matrimony_adnan_67cc880d12d32259d0b13a05']
collection = db["user_profiles"]

# index = collection.create_index({"education": 'text'},default_language='none')
print()
profiles = collection.find({}).to_list(length=None)

# print(len(profiles))

fields_to_check = ["full_name","age","date_of_birth","gender","marital_status","complexion","height","education","maslak_sect","occupation","native_place","residence","location","siblings","father_name","mother_name","preferences","pref_age_range","pref_marital_status","pref_height","pref_complexion","pref_education","pref_work_job","pref_father_occupation","pref_no_of_siblings","pref_native_place","pref_mother_tongue","pref_go_to_dargah","pref_maslak_sect","pref_deendari","pref_location","pref_own_house",
]
total_profiles = len(fields_to_check)
# print(total_profiles)

# 4. Prepare bulk operations
bulk_operations = []

for profile in profiles:
    
    filled_counts = 0
    for field in fields_to_check:
       if profile.get(field) in ["None","",'Nil']:
           filled_counts += 1
        #    print(profile)
        #    print(filled_counts)
    
    profile_completion = (filled_counts / total_profiles) * 100
    print(profile_completion)
    
    # create an Update Operation
    bulk_operations.append(
        UpdateOne(
            {"_id": profile['_id']},
            {"$set": {"profile_completion": int(profile_completion)}}
            )
        )
    

# Execute all updates Once
if bulk_operations:
    result = collection.bulk_write(bulk_operations)
    print(f"Modified {result.modified_count} documents.")
    