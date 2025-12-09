import cv2
import asyncio
from database import (
    users_collection, 
    areas_collection, 
    loitering_collection, # NEW IMPORT
    hash_password, 
    verify_password, 
    area_helper, 
    is_valid_objectid
)
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pd import RealTimePersonDetector 
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Union
from bson import ObjectId
import time

app = FastAPI()

VIDEO_PATH = "vid2.mp4"
stop_stream = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL CACHES ---
active_restricted_areas_cache = []
active_loitering_areas_cache = [] # NEW CACHE

async def update_area_cache():
    """
    Refreshes both restricted and loitering area caches from the database.
    """
    global active_restricted_areas_cache, active_loitering_areas_cache
    
    print("🔄 Refreshing area caches...")
    
    # Fetch Restricted
    r_cursor = areas_collection.find({"is_active": True})
    active_restricted_areas_cache = [area_helper(area) for area in await r_cursor.to_list(length=1000)]
    
    # Fetch Loitering
    l_cursor = loitering_collection.find({"is_active": True})
    active_loitering_areas_cache = [area_helper(area) for area in await l_cursor.to_list(length=1000)]
    
    print(f"✅ Cache updated. Restricted: {len(active_restricted_areas_cache)}, Loitering: {len(active_loitering_areas_cache)}")

@app.on_event("startup")
async def startup_event():
    await update_area_cache()

print("🚀 Initializing YOLO model...")
detector = RealTimePersonDetector(performance_mode='performance')
print("✅ YOLO model ready!")

# --- AUTH ENDPOINTS ---

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/auth/register")
async def register_user(user: UserRegister):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = hash_password(user.password)
    user_doc = {
        "email": user.email,
        "hashed_password": hashed_password,
        "name": user.name,
        "created_at": time.time()
    }
    await users_collection.insert_one(user_doc)
    return {"message": "User registered successfully", "email": user.email}

@app.post("/auth/login")
async def login_user(user: UserLogin):
    user_doc = await users_collection.find_one({"email": user.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(user.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user": user_doc["email"]}

# --- CORE LOGIC: FRAME GENERATION ---

async def generate_frames(request: Request):
    global stop_stream
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / fps

    print("🎥 Starting stream...")
    try:
        while not stop_stream:
            if await request.is_disconnected():
                print("🚪 Client disconnected.")
                break

            # Live access to caches
            restricted_zones = active_restricted_areas_cache
            loitering_zones = active_loitering_areas_cache # Available for detection logic

            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            detections = detector.detect_persons(frame)
            persons = detector.track_persons(detections, frame)
            frame = detector.draw_detections(frame, persons)

            # Check Restricted Area Breaches
            breaches = detector.detect_breaches_with_ids(persons, restricted_zones)
            for pid, zone_name in breaches:
                pdata = persons.get(pid)
                if pdata:
                    detector.create_alert_if_new(frame, pid, pdata['bbox'], zone_name, alert_type="breach")

            # TODO: Add logic here to pass `loitering_zones` to a detector function
            # e.g., detector.detect_loitering(persons, loitering_zones)

            # 2. NEW: Loitering Zone Logic
            # We pass the frame so the detector can crop the image if an alert happens.
            # We pass 10.0 as the threshold seconds.
            detector.detect_loitering(persons, loitering_zones, frame, time_threshold=10.0)

            # 3. [NEW] Running Logic
            detector.detect_running(persons, frame)

            zones_now = {zone for (_, zone) in breaches}
            breach_text = (" | ".join([f"{zone} breached" for zone in zones_now])
                           if zones_now else "All zones clear")
            cv2.putText(frame, breach_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if zones_now else (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            await asyncio.sleep(frame_delay)

    except asyncio.CancelledError:
        print("⚠️ Stream cancelled.")
    finally:
        cap.release()

@app.get("/video_feed")
async def video_feed(request: Request):
    return StreamingResponse(generate_frames(request), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/alerts")
async def get_alerts():
    return JSONResponse(content={"alerts": list(reversed(detector.alerts))})

# --- GENERIC AREA MODEL ---

class AreaModel(BaseModel):
    id: Optional[str] = None 
    name: Optional[str] = None
    type: str = "polygon" 
    points: Optional[List[List[int]]] = [] 

# --- RESTRICTED AREA ENDPOINTS (Existing) ---

@app.post("/restricted-areas/")
async def add_restricted_area(area: AreaModel):
    if area.type == "polygon" and (not area.points or len(area.points) < 3):
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 points.")

    new_area = area.dict(exclude_none=True)
    if "_id" in new_area: del new_area["_id"]
    if "id" in new_area: del new_area["id"]
        
    new_area["is_active"] = True
    new_area["category"] = "restricted" 
    
    if not area.name:
        count = await areas_collection.count_documents({})
        new_area["name"] = f"Restricted Zone {count + 1}"
        
    result = await areas_collection.insert_one(new_area)
    await update_area_cache()
    
    created = await areas_collection.find_one({"_id": result.inserted_id})
    return {"status": "ok", "area": area_helper(created)}

@app.get("/restricted-areas/")
async def list_restricted_areas():
    cursor = areas_collection.find()
    return {"areas": [area_helper(a) for a in await cursor.to_list(1000)]}

@app.delete("/restricted-areas/{area_id}")
async def delete_restricted_area(area_id: str):
    if not is_valid_objectid(area_id): raise HTTPException(400, "Invalid ID")
    res = await areas_collection.delete_one({"_id": ObjectId(area_id)})
    if res.deleted_count == 0: raise HTTPException(404, "Area not found")
    await update_area_cache()
    return {"status": "ok"}

# --- LOITERING AREA ENDPOINTS (New) ---

@app.post("/loitering-areas/")
async def add_loitering_area(area: AreaModel):
    if area.type == "polygon" and (not area.points or len(area.points) < 3):
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 points.")

    new_area = area.dict(exclude_none=True)
    if "_id" in new_area: del new_area["_id"]
    if "id" in new_area: del new_area["id"]
        
    new_area["is_active"] = True
    new_area["category"] = "loitering"
    
    if not area.name:
        count = await loitering_collection.count_documents({})
        new_area["name"] = f"Loitering Zone {count + 1}"
        
    result = await loitering_collection.insert_one(new_area)
    await update_area_cache() # Update cache immediately
    
    created = await loitering_collection.find_one({"_id": result.inserted_id})
    return {"status": "ok", "area": area_helper(created)}

@app.get("/loitering-areas/")
async def list_loitering_areas():
    cursor = loitering_collection.find()
    return {"areas": [area_helper(a) for a in await cursor.to_list(1000)]}

@app.delete("/loitering-areas/{area_id}")
async def delete_loitering_area(area_id: str):
    if not is_valid_objectid(area_id): raise HTTPException(400, "Invalid ID")
    res = await loitering_collection.delete_one({"_id": ObjectId(area_id)})
    if res.deleted_count == 0: raise HTTPException(404, "Area not found")
    await update_area_cache()
    return {"status": "ok"}

# --- TOGGLE ENDPOINTS (Updated for both) ---

@app.patch("/restricted-areas/{area_id}/toggle")
async def toggle_restricted_area(area_id: str):
    if not is_valid_objectid(area_id): raise HTTPException(400, "Invalid ID")
    
    area = await areas_collection.find_one({"_id": ObjectId(area_id)})
    if not area: raise HTTPException(404, "Area not found")
    
    new_state = not area.get("is_active", True)
    await areas_collection.update_one({"_id": ObjectId(area_id)}, {"$set": {"is_active": new_state}})
    await update_area_cache()
    return {"status": "ok", "is_active": new_state}

@app.patch("/loitering-areas/{area_id}/toggle")
async def toggle_loitering_area(area_id: str):
    if not is_valid_objectid(area_id): raise HTTPException(400, "Invalid ID")
    
    area = await loitering_collection.find_one({"_id": ObjectId(area_id)})
    if not area: raise HTTPException(404, "Area not found")
    
    new_state = not area.get("is_active", True)
    await loitering_collection.update_one({"_id": ObjectId(area_id)}, {"$set": {"is_active": new_state}})
    await update_area_cache()
    return {"status": "ok", "is_active": new_state}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)