import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")

# --- Database Connection ---
client = AsyncIOMotorClient(MONGO_URI)
database = client[MONGO_DB_NAME]
users_collection = database.get_collection("users")

# --- Password Hashing Utilities ---

def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt."""
    # Encode password and hash
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a hash."""
    try:
        # Encode inputs and check
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        # Handle cases where the hash might be invalid (e.g., bad format)
        return False