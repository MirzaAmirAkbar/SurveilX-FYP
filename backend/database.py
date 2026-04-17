import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from typing import Dict, Any
from bson import ObjectId

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Database Connection ---
client = AsyncIOMotorClient(MONGO_URI)
database = client[MONGO_DB_NAME]

# Collections
users_collection = database.get_collection("users")
areas_collection = database.get_collection("restricted_areas")
loitering_collection = database.get_collection("loitering_areas") # NEW: Collection for loitering
alerts_collection = database.get_collection("alerts") # NEW: Collection for persistent alerts

# --- Utility Functions ---

def area_helper(area: dict) -> dict:
    return {
        "id": str(area["_id"]),
        "name": area.get("name", "Unnamed Area"),
        "type": area["type"],
        # We can store the category in the DB to distinguish later if needed
        "category": area.get("category", "security"), 
        "points": area.get("points", []),
        "center": area.get("center", None),
        "radii": area.get("radii", None),
        "is_active": area.get("is_active", True)
    }

def is_valid_objectid(id_str: str) -> bool:
    """Checks if a string is a valid MongoDB ObjectId format."""
    return ObjectId.is_valid(id_str)

# --- Password Hashing Utilities ---

def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False