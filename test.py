from datetime import datetime
def calculate_age_from_dob(dob_str: str) -> int:
    """
    Calculate age from date of birth string (format: YYYY-MM-DD).
    """
    dob = datetime.strptime(dob_str, "%d-%m-%Y")
    today = datetime.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age

age = calculate_age_from_dob("22-10-2001")
print(age)