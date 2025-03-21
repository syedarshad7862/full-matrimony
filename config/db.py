from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
# connect = MongoClient(MONGO_URI)
client = AsyncIOMotorClient(MONGO_URI)
agents_db = client["matrimony_agents"] # Common DB for storing agents
# user_collection = db["user"]