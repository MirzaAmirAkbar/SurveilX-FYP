import sys      # <-- NEW
import os       # <-- NEW

import cv2
import asyncio
from database import (
    users_collection, 
    areas_collection, 
    loitering_collection, 
    hash_password, 
    verify_password, 
    area_helper, 
    is_valid_objectid
)
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pd import RealTimePersonDetector 

from weapon_detection import WeaponDetector # <-- NEW IMPORT
import base64                               # <-- NEW IMPORT (if not present)

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Union
from bson import ObjectId
import time

import math

# --- NEW IMPORTS FOR RF-DETR ---
import numpy as np
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# --- NEW GLOBALS FOR ITEM TRACKING ---
STATIONARY_THRESHOLD_FRAMES = 90
MOVEMENT_TOLERANCE_PIXELS = 15
OWNER_ASSIGNMENT_DISTANCE = 120  # Pixels: How close a person must be to claim an item
ABANDONED_DISTANCE_PIXELS = 180  # Pixels: How far the owner must be to trigger abandoned
ITEM_PATIENCE_FRAMES = 30        # Frames to remember an item if it flickers out
item_states = {}                 # Tracks {track_id: {center, owner, frames_stationary, etc.}}

def get_center_distance(box1, box2):
    """Calculates the distance between the centers of two bounding boxes."""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.hypot(cx1 - cx2, cy1 - cy2)


# --- ADD SHOPLIFTING TO PYTHON PATH ---
# This tells Python to also look inside the Shoplifting folder when resolving imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Shoplifting'))
# --- ADD THESE IMPORTS ---
from Shoplifting.data_pipeline import ShopliftingDetector
from Shoplifting.config import WEIGHT_FILE, SLIDING_WINDOW_STEP, INPUT_FRAMES
import threading # For non-blocking inference

# --- INITIALIZE SHOPLIFTING DETECTOR ---
print("🛒 Initializing Shoplifting Detector...")
# Ensure the path to the weights file is correct relative to main.py
shoplifting_detector = ShopliftingDetector(weights_path="Shoplifting/weights/weights_at_epoch_8_75___good.h5")
shoplifting_detector.load_model()
print("✅ Shoplifting Detector ready!")

# --- NEW GLOBALS FOR SHOPLIFTING ---
shoplifting_frame_buffer = []          # Stores the last 64 frames
shoplifting_alert_cooldown_frames = 0  # Keeps the warning on screen for a few seconds
shoplifting_current_status = None      # Holds the latest text to draw on screen
is_shoplifting_inferencing = False     # Prevents overlapping inference threads

app = FastAPI()

VIDEO_PATH = "test_videos/shoplifting-43.mp4"
stop_stream = False

last_weapon_alert_time = 0.0      # <-- NEW GLOBAL
WEAPON_ALERT_COOLDOWN = 5.0       # <-- NEW GLOBAL (Seconds between alerts)

# --- NEW GLOBALS FOR WEAPON STATE ---
WEAPON_FRAME_THRESHOLD = 3  # Weapon must be seen for 5 consecutive frames
weapon_states = {}          # Tracks {pid: {'frames': 0, 'alerted': False}}

def get_ioa(box1, box2):
    """
    Calculates Intersection over Area (IoA) of box1 (weapon) relative to itself.
    This works better than IoU because a small weapon inside a large person
    bbox will have low IoU, but 100% IoA.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])

    return inter_area / box1_area if box1_area > 0 else 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL CACHES ---
active_restricted_areas_cache = []
active_loitering_areas_cache = []

async def update_area_cache():
    global active_restricted_areas_cache, active_loitering_areas_cache
    
    print("🔄 Refreshing area caches...")
    r_cursor = areas_collection.find({"is_active": True})
    active_restricted_areas_cache = [area_helper(area) for area in await r_cursor.to_list(length=1000)]
    
    l_cursor = loitering_collection.find({"is_active": True})
    active_loitering_areas_cache = [area_helper(area) for area in await l_cursor.to_list(length=1000)]
    
    print(f"✅ Cache updated. Restricted: {len(active_restricted_areas_cache)}, Loitering: {len(active_loitering_areas_cache)}")

@app.on_event("startup")
async def startup_event():
    await update_area_cache()

print("🚀 Initializing YOLO model & Face Recognition...")
# Initializing with default Milvus params
detector = RealTimePersonDetector(performance_mode='performance', milvus_host="localhost", milvus_port="19530")
print("✅ Detector ready!")


print("🔫 Initializing Weapon Detector...")
# Initialize the weapon detector. Ensure the model file is in the correct directory.
weapon_detector = WeaponDetector(model_path='w1.pt', conf_threshold=0.6)


# --- [NEW] INITIALIZE RF-DETR ITEM DETECTOR ---
print("🎒 Initializing RF-DETR Item Detector...")
rfdetr_model = RFDETRBase()
rfdetr_model.optimize_for_inference()

item_tracker = sv.ByteTrack()
TARGET_CATEGORIES = {'backpack', 'umbrella', 'handbag', 'suitcase', 'bottle', 'cell phone'}

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
print("✅ Item Detector ready!")

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

def run_shoplifting_inference(frames_to_process, current_time):
    global shoplifting_alert_cooldown_frames, shoplifting_current_status, is_shoplifting_inferencing
    
    try:
        # Run prediction on the 64-frame sequence
        bag, clothes, normal, is_theft, event_type = shoplifting_detector.predict_frame_set(frames_to_process)
        
        theft_conf = bag + clothes
        if is_theft:
            shoplifting_current_status = f"THEFT DETECTED: {event_type} ({theft_conf*100:.1f}%)"
            shoplifting_alert_cooldown_frames = 60 # Show alert for approx 2 seconds (at 30fps)
            
            # --- Generate the API Alert ---
            # Grab the last frame of the sequence for the thumbnail
            latest_frame = frames_to_process[-1]
            _, buf = cv2.imencode('.jpg', latest_frame)
            img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            
            alert = {
                'personId': "Unknown", # The model analyzes the whole scene, not a specific ID
                'zone': "Global",
                'type': f"shoplifting_{event_type.lower()}",
                'timestamp': current_time,
                'imageB64': img_b64,
            }
            # Note: Assuming 'detector' (RealTimePersonDetector) is accessible globally here
            detector.alerts.append(alert)
            print(f"⚠️ ALERT: Shoplifting Detected! Type: {event_type}, Confidence: {theft_conf*100:.1f}%")
            
    except Exception as e:
        print(f"[!] Shoplifting inference error: {e}")
    finally:
        is_shoplifting_inferencing = False



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

            restricted_zones = active_restricted_areas_cache
            loitering_zones = active_loitering_areas_cache 

            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # --- [NEW] SHOPLIFTING FRAME BUFFERING ---
            global shoplifting_frame_buffer, is_shoplifting_inferencing, shoplifting_alert_cooldown_frames, shoplifting_current_status
            
            # Add a copy of the raw frame to the buffer
            shoplifting_frame_buffer.append(frame.copy())
            
            # Maintain a maximum of 64 frames
            if len(shoplifting_frame_buffer) > INPUT_FRAMES:
                shoplifting_frame_buffer.pop(0)

            # Trigger inference every SLIDING_WINDOW_STEP frames (e.g., every 32 frames)
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if len(shoplifting_frame_buffer) == INPUT_FRAMES and (current_frame_pos % SLIDING_WINDOW_STEP == 0):
                if not is_shoplifting_inferencing:
                    is_shoplifting_inferencing = True
                    frames_copy = list(shoplifting_frame_buffer) # Copy buffer to prevent mutation
                    current_time = time.time()
                    # Start background thread
                    threading.Thread(
                        target=run_shoplifting_inference, 
                        args=(frames_copy, current_time), 
                        daemon=True
                    ).start()

                
            detections = detector.detect_persons(frame)
            persons = detector.track_persons(detections, frame)

            # --- [NEW] FACE RECOGNITION STEP ---
            #detector.recognize_identities(frame, persons)
            # -----------------------------------

            #frame = detector.draw_detections(frame, persons)
            """
            # --- [NEW] WEAPON DETECTION & VISUALIZATION ---
            weapon_detections = weapon_detector.detect_weapons(frame)
            frame = weapon_detector.draw_detections(frame, weapon_detections)

            # --- [NEW] WEAPON ALERT LOGIC ---
            current_time = time.time()
            if weapon_detections and (1):
                #last_weapon_alert_time = current_time
                
                # Capture the first detected weapon for the alert thumbnail
                x1, y1, x2, y2 = weapon_detections[0]['bbox']
                h_frame, w_frame = frame.shape[:2]
                crop = frame[max(0, y1):min(h_frame, y2), max(0, x1):min(w_frame, x2)]
                
                img_b64 = ""
                if crop.size > 0:
                    _, buf = cv2.imencode('.jpg', crop)
                    img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                    
                weapon_alert = {
                    'personId': "N/A", # Weapons are not tied to a specific ID yet
                    'zone': "Global",
                    'type': "weapon_detected",
                    'timestamp': current_time,
                    'imageB64': img_b64,
                }
                detector.alerts.append(weapon_alert)
                print("⚠️ ALERT: Weapon detected and logged!")
            """

            # --- [UPDATED] RF-DETR ITEM DETECTION & VISUALIZATION ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            item_detections = rfdetr_model.predict(rgb_frame, threshold=0.45)

            mask = np.array([COCO_CLASSES[class_id] in TARGET_CATEGORIES for class_id in item_detections.class_id], dtype=bool)
            item_detections = item_detections[mask]
            item_detections = item_tracker.update_with_detections(detections=item_detections)

            custom_labels = []
            current_item_ids = set()

            for i in range(len(item_detections)):
                bbox = item_detections.xyxy[i]
                track_id = item_detections.tracker_id[i]
                class_id = item_detections.class_id[i]
                class_name = COCO_CLASSES[class_id]
                
                current_item_ids.add(track_id)
                current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                # Initialize state for new objects
                if track_id not in item_states:
                    item_states[track_id] = {
                        "center": current_center,
                        "frames_stationary": 0,
                        "owner_id": None,
                        "status": "Tracking",
                        "alerted": False,
                        "lost_frames": 0,
                    }

                state = item_states[track_id]
                state["lost_frames"] = 0  # Reset flicker patience

                # 1. Calculate Movement
                prev_center = state["center"]
                distance_moved = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])

                if distance_moved < MOVEMENT_TOLERANCE_PIXELS:
                    state["frames_stationary"] += 1
                else:
                    state["frames_stationary"] = 0
                    state["center"] = current_center
                    state["status"] = "Tracking"

                # 2. Find Closest Person (Potential Owner)
                closest_pid = None
                min_dist = float('inf')
                for pid, pdata in persons.items():
                    dist = get_center_distance(bbox, pdata['bbox'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pid = pid

                # Assign owner if the item is recently moving and a person is very close
                if state["frames_stationary"] < 15 and min_dist < OWNER_ASSIGNMENT_DISTANCE:
                    state["owner_id"] = closest_pid

                # 3. Abandoned Logic
                if state["frames_stationary"] > STATIONARY_THRESHOLD_FRAMES:
                    # Check how far the assigned owner is (if they are still in frame)
                    owner_dist = float('inf')
                    if state["owner_id"] is not None and state["owner_id"] in persons:
                        owner_dist = get_center_distance(bbox, persons[state["owner_id"]]['bbox'])
                    
                    # If BOTH the assigned owner is far AND no other person is close by
                    if min_dist > ABANDONED_DISTANCE_PIXELS and owner_dist > ABANDONED_DISTANCE_PIXELS:
                        state["status"] = "⚠️ ABANDONED"
                        
                        if not state["alerted"]:
                            # Generate Alert
                            x1, y1, x2, y2 = map(int, bbox)
                            h_f, w_f = frame.shape[:2]
                            crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                            img_b64 = ""
                            if crop.size > 0:
                                _, buf = cv2.imencode('.jpg', crop)
                                img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

                            alert = {
                                'personId': state["owner_id"] if state["owner_id"] else "Unknown",
                                'zone': "Global",
                                'type': "abandoned_object",
                                'timestamp': time.time(),
                                'imageB64': img_b64,
                            }
                            detector.alerts.append(alert)
                            state["alerted"] = True
                            print(f"⚠️ ALERT: Abandoned {class_name} (ID: {track_id}) detected!")
                    else:
                        # Someone is standing near it, so it's not abandoned yet
                        state["status"] = "Stationary"
                        state["alerted"] = False 

                # Format label for the bounding box
                owner_text = f"Owner:{state['owner_id']}" if state['owner_id'] else "Owner:None"
                custom_labels.append(f"#{track_id} {class_name} | {owner_text} | {state['status']}")

            # 4. Handle Flickering and Cleanup
            for t_id in list(item_states.keys()):
                if t_id not in current_item_ids:
                    item_states[t_id]["lost_frames"] += 1
                    # If the object is missing for more than ITEM_PATIENCE_FRAMES, forget it
                    if item_states[t_id]["lost_frames"] > ITEM_PATIENCE_FRAMES:
                        del item_states[t_id]

            # Annotate frame
            frame = box_annotator.annotate(scene=frame, detections=item_detections)
            if len(item_detections) > 0:
                frame = label_annotator.annotate(scene=frame, detections=item_detections, labels=custom_labels)
            # ----------------------------------------------------



            # --- [NEW] WEAPON DETECTION & VISUALIZATION ---
            weapon_detections = weapon_detector.detect_weapons(frame)
            
            # 1. Map Weapons to Persons
            current_armed_pids = set()
            for w_det in weapon_detections:
                best_pid = None
                best_ioa = 0.0

                for pid, pdata in persons.items():
                    ioa = get_ioa(w_det['bbox'], pdata['bbox'])
                    # If at least 30% of the weapon is inside the person's bounding box
                    if ioa > best_ioa and ioa > 0.3: 
                        best_ioa = ioa
                        best_pid = pid

                if best_pid is not None:
                    current_armed_pids.add(best_pid)
                    w_det['person_id'] = best_pid
                else:
                    w_det['person_id'] = "Unknown"

            # 2. Temporal Smoothing & Alerts
            for pid in list(persons.keys()):
                # Initialize state for new persons
                if pid not in weapon_states:
                    weapon_states[pid] = {'frames': 0, 'alerted': False}

                if pid in current_armed_pids:
                    # Increment counter if weapon is detected
                    weapon_states[pid]['frames'] += 1

                    # Trigger alert if threshold met and not already alerted
                    if weapon_states[pid]['frames'] >= WEAPON_FRAME_THRESHOLD and not weapon_states[pid]['alerted']:
                        pdata = persons[pid]
                        x1, y1, x2, y2 = pdata['bbox']
                        h_f, w_f = frame.shape[:2]
                        
                        # Crop the PERSON holding the weapon for the alert thumbnail
                        crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                        
                        img_b64 = ""
                        if crop.size > 0:
                            _, buf = cv2.imencode('.jpg', crop)
                            img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

                        weapon_alert = {
                            'personId': pid,
                            'zone': "",
                            'type': "weapon_detected",
                            'timestamp': time.time(),
                            'imageB64': img_b64,
                        }
                        detector.alerts.append(weapon_alert)
                        print(f"⚠️ ALERT: Weapon confirmed for Person {pid}!")
                        
                        # Lock the alert so it doesn't spam
                        weapon_states[pid]['alerted'] = True
                else:
                    # Decay the frame count if weapon disappears (tolerates brief flicker)
                    weapon_states[pid]['frames'] = max(0, weapon_states[pid]['frames'] - 1)
                    
                    # Reset the alert lock only if they fully drop/put away the weapon
                    if weapon_states[pid]['frames'] == 0:
                        weapon_states[pid]['alerted'] = False

            # Cleanup state for lost tracks to prevent memory leaks
            for pid in list(weapon_states.keys()):
                if pid not in persons and weapon_states[pid]['frames'] == 0:
                    del weapon_states[pid]

            # 3. Draw Detections
            frame = weapon_detector.draw_detections(frame, weapon_detections)

            # Check Restricted Area Breaches
            breaches = detector.detect_breaches_with_ids(persons, restricted_zones)
            for pid, zone_name in breaches:
                pdata = persons.get(pid)
                if pdata:
                    detector.create_alert_if_new(frame, pid, pdata['bbox'], zone_name, alert_type="breach")

            # Loitering Logic
            detector.detect_loitering(persons, loitering_zones, frame, time_threshold=10.0)

            # Running Logic (Optional - commented out in your original code)
            # detector.detect_running(persons, frame)

            zones_now = {zone for (_, zone) in breaches}
            breach_text = (" | ".join([f"{zone} breached" for zone in zones_now])
                           if zones_now else "All zones clear")
            cv2.putText(frame, breach_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if zones_now else (0, 255, 0), 2)
            
            # --- [NEW] DRAW SHOPLIFTING ALERTS ---
            if shoplifting_alert_cooldown_frames > 0:
                # Draw a prominent red warning box at the top right
                cv2.rectangle(frame, (frame.shape[1] - 450, 10), (frame.shape[1] - 10, 60), (0, 0, 255), -1)
                cv2.putText(frame, shoplifting_current_status, (frame.shape[1] - 440, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                shoplifting_alert_cooldown_frames -= 1

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

# --- RESTRICTED AREA ENDPOINTS ---

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

# --- LOITERING AREA ENDPOINTS ---

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
    await update_area_cache() 
    
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

# --- TOGGLE ENDPOINTS ---

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