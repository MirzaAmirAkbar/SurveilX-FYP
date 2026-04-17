# ==========================================
# SECTION 1: IMPORTS
# ==========================================

# 1.1 Standard Library Imports
import os
import sys
import time
import math
import base64
import asyncio
import threading
from typing import Optional, List

# 1.2 FastAPI & Web Imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 1.3 Database & Data Validation Imports (Pydantic / MongoDB)
from pydantic import BaseModel, EmailStr
from bson import ObjectId
from database import (
    users_collection, 
    areas_collection, 
    loitering_collection, 
    hash_password, 
    verify_password, 
    area_helper, 
    is_valid_objectid
)

# 1.4 Computer Vision & Machine Learning Imports
import cv2
import numpy as np
import supervision as sv
from cjm_byte_track.core import BYTETracker

# 1.5 Custom Application Modules
from pd import RealTimePersonDetector 
from weapon_detection import WeaponDetector 
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

# --- Path adjustment for Shoplifting module ---
# This ensures Python can locate modules inside the 'Shoplifting' directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'Shoplifting'))
from Shoplifting.data_pipeline import ShopliftingDetector
from Shoplifting.config import WEIGHT_FILE, SLIDING_WINDOW_STEP, INPUT_FRAMES


# ==========================================
# SECTION 2: GLOBAL CONSTANTS & STATE
# ==========================================

# --- General Stream Settings ---
VIDEO_PATH = "test_videos/shoplifting-36.mp4"
stop_stream = False
# Add a flag to track if we need to reset trackers when switching videos
reset_trackers_flag = False

# --- Item Tracking Constants (RF-DETR) ---
STATIONARY_THRESHOLD_FRAMES = 90
MOVEMENT_TOLERANCE_PIXELS = 15
OWNER_ASSIGNMENT_DISTANCE = 120  # Pixels: How close a person must be to claim an item
ABANDONED_DISTANCE_PIXELS = 180  # Pixels: How far the owner must be to trigger abandoned
ITEM_PATIENCE_FRAMES = 30        # Frames to remember an item if it flickers out
TARGET_CATEGORIES = {'backpack', 'handbag', 'suitcase'}

# Dictionary to track state of detected items over time
# Format: {track_id: {center, owner, frames_stationary, status, alerted, lost_frames}}
item_states = {}                 

# --- Shoplifting Detection Constants ---
shoplifting_frame_buffer = []          # Stores the last 64 frames for sequence analysis
shoplifting_alert_cooldown_frames = 0  # Keeps the warning on screen for a few seconds
shoplifting_current_status = None      # Holds the latest text to draw on screen
is_shoplifting_inferencing = False     # Prevents overlapping inference threads

# --- Weapon Detection Constants ---
WEAPON_ALERT_COOLDOWN = 5.0      
WEAPON_FRAME_THRESHOLD = 3       # Weapon must be seen for 3 consecutive frames to trigger alert

# Dictionary to track weapon possession state per person
# Format: {pid: {'frames': 0, 'alerted': False}}
weapon_states = {}               

# --- Caching for Database Zones ---
# Stores active zones in memory to avoid querying the DB every frame
active_restricted_areas_cache = []
active_loitering_areas_cache = []


# ==========================================
# SECTION 3: HELPER FUNCTIONS
# ==========================================

def get_center_distance(box1, box2):
    """
    Calculates the Euclidean distance between the centers of two bounding boxes.
    Used to determine proximity between people and objects.
    """
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.hypot(cx1 - cx2, cy1 - cy2)

def get_ioa(box1, box2):
    """
    Calculates Intersection over Area (IoA) of box1 relative to itself.
    Better than IoU for cases where a small box (weapon) is entirely inside a large box (person).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])

    return inter_area / box1_area if box1_area > 0 else 0


# ==========================================
# SECTION 4: ML MODEL INITIALIZATION
# ==========================================
# Models are loaded into memory globally upon server start to prevent latency

print("🛒 Initializing Shoplifting Detector...")
shoplifting_detector = ShopliftingDetector(weights_path="Shoplifting/weights/weights_at_epoch_8_75___good.h5")
shoplifting_detector.load_model()
print("✅ Shoplifting Detector ready!")

print("🚀 Initializing YOLO model & Face Recognition...")
detector = RealTimePersonDetector(performance_mode='performance', milvus_host="localhost", milvus_port="19530")
print("✅ Detector ready!")

print("🔫 Initializing Weapon Detector...")
weapon_detector = WeaponDetector(model_path='w1.pt', conf_threshold=0.6)
print("✅ Weapon Detector ready!")

print("🎒 Initializing RF-DETR Item Detector...")
rfdetr_model = RFDETRBase()
rfdetr_model.optimize_for_inference()
item_tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --- CHANGED: Force a single Orange color for all items ---
bag_color = sv.Color.from_hex("#008080")
box_annotator = sv.BoxAnnotator(color=bag_color)
label_annotator = sv.LabelAnnotator(color=bag_color)

print("✅ Item Detector ready!")


# ==========================================
# SECTION 5: FASTAPI SETUP & LIFECYCLE
# ==========================================

app = FastAPI(title="Surveillance Backend API")

# Add CORS middleware to allow requests from frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def update_area_cache():
    """Fetches active restricted and loitering areas from the database and updates RAM cache."""
    global active_restricted_areas_cache, active_loitering_areas_cache
    
    print("🔄 Refreshing area caches...")
    r_cursor = areas_collection.find({"is_active": True})
    active_restricted_areas_cache = [area_helper(area) for area in await r_cursor.to_list(length=1000)]
    
    l_cursor = loitering_collection.find({"is_active": True})
    active_loitering_areas_cache = [area_helper(area) for area in await l_cursor.to_list(length=1000)]
    
    print(f"✅ Cache updated. Restricted: {len(active_restricted_areas_cache)}, Loitering: {len(active_loitering_areas_cache)}")

@app.on_event("startup")
async def startup_event():
    """Runs when the FastAPI server starts."""
    await update_area_cache()


# ==========================================
# SECTION 6: BACKGROUND TASKS
# ==========================================

def run_shoplifting_inference(frames_to_process, current_time):
    """
    Executes shoplifting inference on a sequence of frames in a separate thread.
    This prevents the main video stream from freezing during heavy computation.
    """
    global shoplifting_alert_cooldown_frames, shoplifting_current_status, is_shoplifting_inferencing
    
    try:
        # Run 3D CNN prediction on the 64-frame sequence
        bag, clothes, normal, is_theft, event_type = shoplifting_detector.predict_frame_set(frames_to_process)
        theft_conf = bag + clothes
        
        if is_theft:
            shoplifting_current_status = f"THEFT DETECTED: {event_type} ({theft_conf*100:.1f}%)"
            shoplifting_alert_cooldown_frames = 60 # Display alert for ~2 seconds at 30fps
            
            # Generate Base64 thumbnail from the last frame for the API response
            latest_frame = frames_to_process[-1]
            _, buf = cv2.imencode('.jpg', latest_frame)
            img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            
            # Append alert to global detector log
            alert = {
                'personId': "Unknown", 
                'zone': "Global",
                'type': f"shoplifting_{event_type.lower()}",
                'timestamp': current_time,
                'imageB64': img_b64,
            }
            detector.alerts.append(alert)
            print(f"⚠️ ALERT: Shoplifting Detected! Type: {event_type}, Confidence: {theft_conf*100:.1f}%")
            
    except Exception as e:
        print(f"[!] Shoplifting inference error: {e}")
    finally:
        # Release the lock so the next inference cycle can begin
        is_shoplifting_inferencing = False


# ==========================================
# SECTION 7: CORE VIDEO STREAMING LOGIC
# ==========================================

async def generate_frames(request: Request):
    """
    Main loop that reads video frames, runs all AI models, processes logic, 
    and yields JPEG byte streams to the client.
    """
    global stop_stream, VIDEO_PATH, reset_trackers_flag
    
    active_path = VIDEO_PATH
    cap = cv2.VideoCapture(active_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1 / fps

    print("🎥 Starting stream...")
    try:
        while not stop_stream:
            # Check for client disconnect to prevent memory leaks
            if await request.is_disconnected():
                print("🚪 Client disconnected.")
                break

            global shoplifting_frame_buffer, is_shoplifting_inferencing, shoplifting_alert_cooldown_frames, shoplifting_current_status, item_tracker

            # --- HOT SWAP LOGIC ---
            if reset_trackers_flag:
                print(f"🔄 Switching feed to {VIDEO_PATH}")
                cap.release()
                
                # 1. Clear Shoplifting and Item/Weapon buffers
                shoplifting_frame_buffer.clear()
                item_states.clear()
                weapon_states.clear()
                
                # 2. Reset the Item Tracker (Supervision library)
                item_tracker = sv.ByteTrack()
                
                # 3. Reset the Person Tracker (CJM ByteTrack library)
                detector.tracker = BYTETracker(track_thresh=0.5, track_buffer=90, match_thresh=0.7, frame_rate=30)
                
                # 4. Clear Person Detector tracking memories so IDs and zones don't bleed across videos
                detector.stable_id_features.clear()
                detector.byte_to_stable_id.clear()
                detector.stable_id_last_seen.clear()
                detector.stable_id_positions.clear()
                detector.running_cooldowns.clear()
                detector.loitering_state.clear()
                detector.person_identities.clear()
                detector.current_breaching_ids.clear()
                detector.confirmed_loiterers.clear()
                detector.active_breaches.clear()

                # ---> ADD THESE TWO LINES <---
                detector.next_stable_id = 1
                detector.frame_index = 0
                
                active_path = VIDEO_PATH
                cap = cv2.VideoCapture(active_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_delay = 1 / fps
                reset_trackers_flag = False

            # Fetch current zones from cache
            restricted_zones = active_restricted_areas_cache
            loitering_zones = active_loitering_areas_cache 

            # Read frame from video source
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video if it ends
                continue

            # ----------------------------------------------------
            # STEP 1: SHOPLIFTING BUFFERING & INFERENCE TRIGGER
            # ----------------------------------------------------
           
            
            # Maintain a sliding window buffer of the last INPUT_FRAMES (e.g., 64)
            shoplifting_frame_buffer.append(frame.copy())
            if len(shoplifting_frame_buffer) > INPUT_FRAMES:
                shoplifting_frame_buffer.pop(0)

            # Trigger inference periodically based on SLIDING_WINDOW_STEP
            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if len(shoplifting_frame_buffer) == INPUT_FRAMES and (current_frame_pos % SLIDING_WINDOW_STEP == 0):
                if not is_shoplifting_inferencing:
                    is_shoplifting_inferencing = True
                    frames_copy = list(shoplifting_frame_buffer) 
                    current_time = time.time()
                    
                    # Offload inference to background thread
                    threading.Thread(
                        target=run_shoplifting_inference, 
                        args=(frames_copy, current_time), 
                        daemon=True
                    ).start()

            # ----------------------------------------------------
            # STEP 2: PERSON DETECTION & TRACKING
            # ----------------------------------------------------
            detections = detector.detect_persons(frame)
            persons = detector.track_persons(detections, frame)
            
            #detector.recognize_identities(frame, persons)
            #detector.draw_detections(frame, persons)

            # ----------------------------------------------------
            # STEP 3: ITEM DETECTION & ABANDONMENT LOGIC
            # ----------------------------------------------------
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            item_detections = rfdetr_model.predict(rgb_frame, threshold=0.45)

            # Filter out non-target categories (e.g., keep only bags, phones)
            mask = np.array([COCO_CLASSES[class_id] in TARGET_CATEGORIES for class_id in item_detections.class_id], dtype=bool)
            item_detections = item_detections[mask]
            item_detections = item_tracker.update_with_detections(detections=item_detections)

            custom_labels = []
            current_item_ids = set()

            # Process each detected item
            for i in range(len(item_detections)):
                bbox = item_detections.xyxy[i]
                track_id = item_detections.tracker_id[i]
                
                # Override the specific COCO class with our unified category
                class_name = "bag"
                
                current_item_ids.add(track_id)
                current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

                # Initialize tracking state if item is newly detected
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
                state["lost_frames"] = 0  

                # Calculate movement to determine if stationary
                prev_center = state["center"]
                distance_moved = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])

                if distance_moved < MOVEMENT_TOLERANCE_PIXELS:
                    state["frames_stationary"] += 1
                else:
                    state["frames_stationary"] = 0
                    state["center"] = current_center
                    state["status"] = "Tracking"

                # Map items to owners based on proximity
                closest_pid = None
                min_dist = float('inf')
                for pid, pdata in persons.items():
                    dist = get_center_distance(bbox, pdata['bbox'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pid = pid

                # Assign ownership if the item was recently moving and someone is close
                if state["frames_stationary"] < 15 and min_dist < OWNER_ASSIGNMENT_DISTANCE:
                    state["owner_id"] = closest_pid

                # Check for Abandonment
                if state["frames_stationary"] > STATIONARY_THRESHOLD_FRAMES:
                    owner_dist = float('inf')
                    if state["owner_id"] is not None and state["owner_id"] in persons:
                        owner_dist = get_center_distance(bbox, persons[state["owner_id"]]['bbox'])
                    
                    # Trigger alert if BOTH the owner is far away AND no other person is near
                    if min_dist > ABANDONED_DISTANCE_PIXELS and owner_dist > ABANDONED_DISTANCE_PIXELS:
                        state["status"] = "⚠️ ABANDONED"
                        
                        if not state["alerted"]:
                            # Crop item image for alert
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
                        state["status"] = "Stationary"
                        state["alerted"] = False 

                owner_text = f"Owner:{state['owner_id']}" if state['owner_id'] else "Owner:None"
                custom_labels.append(f"{class_name} | {owner_text} | {state['status']}")

            # Cleanup old items that have disappeared for too long
            for t_id in list(item_states.keys()):
                if t_id not in current_item_ids:
                    item_states[t_id]["lost_frames"] += 1
                    if item_states[t_id]["lost_frames"] > ITEM_PATIENCE_FRAMES:
                        del item_states[t_id]

            # Draw Item Annotations
            frame = box_annotator.annotate(scene=frame, detections=item_detections)
            if len(item_detections) > 0:
                frame = label_annotator.annotate(scene=frame, detections=item_detections, labels=custom_labels)


            # ----------------------------------------------------
            # STEP 4: WEAPON DETECTION LOGIC
            # ----------------------------------------------------
            weapon_detections = weapon_detector.detect_weapons(frame)
            current_armed_pids = set()
            
            # Map weapons to people using Intersection over Area (IoA)
            for w_det in weapon_detections:
                best_pid = None
                best_ioa = 0.0

                for pid, pdata in persons.items():
                    ioa = get_ioa(w_det['bbox'], pdata['bbox'])
                    if ioa > best_ioa and ioa > 0.3:  # At least 30% overlap
                        best_ioa = ioa
                        best_pid = pid

                if best_pid is not None:
                    current_armed_pids.add(best_pid)
                    w_det['person_id'] = best_pid
                else:
                    w_det['person_id'] = "Unknown"

            # Check consecutive frame thresholds for alerts
            for pid in list(persons.keys()):
                if pid not in weapon_states:
                    weapon_states[pid] = {'frames': 0, 'alerted': False}

                if pid in current_armed_pids:
                    weapon_states[pid]['frames'] += 1

                    if weapon_states[pid]['frames'] >= WEAPON_FRAME_THRESHOLD and not weapon_states[pid]['alerted']:
                        # Generate weapon alert
                        pdata = persons[pid]
                        x1, y1, x2, y2 = pdata['bbox']
                        h_f, w_f = frame.shape[:2]
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
                        weapon_states[pid]['alerted'] = True
                else:
                    weapon_states[pid]['frames'] = max(0, weapon_states[pid]['frames'] - 1)
                    if weapon_states[pid]['frames'] == 0:
                        weapon_states[pid]['alerted'] = False

            # Memory cleanup for weapon tracking
            for pid in list(weapon_states.keys()):
                if pid not in persons and weapon_states[pid]['frames'] == 0:
                    del weapon_states[pid]

            frame = weapon_detector.draw_detections(frame, weapon_detections)

            # ----------------------------------------------------
            # STEP 5: ZONE BREACH & LOITERING CHECKS
            # ----------------------------------------------------
            breaches = detector.detect_breaches_with_ids(persons, restricted_zones)
            for pid, zone_name in breaches:
                pdata = persons.get(pid)
                if pdata:
                    detector.create_alert_if_new(frame, pid, pdata['bbox'], zone_name, alert_type="breach")

            detector.detect_loitering(persons, loitering_zones, frame, time_threshold=10.0)


            # ----------------------------------------------------
            # STEP 6: YIELD STREAM
            # ----------------------------------------------------
            _, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            await asyncio.sleep(frame_delay)

    except asyncio.CancelledError:
        print("⚠️ Stream cancelled.")
    finally:
        cap.release()


# ==========================================
# SECTION 8: AUTH & MEDIA ENDPOINTS
# ==========================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class VideoSwitchRequest(BaseModel):
    filename: str

@app.get("/api/videos")
async def list_videos():
    """Scans the test_videos directory and returns available files."""
    videos = []
    if os.path.exists("test_videos"):
        for f in os.listdir("test_videos"):
            if f.endswith(('.mp4', '.avi', '.mov')):
                videos.append({"id": f, "name": f})
    return JSONResponse(content={"videos": videos})

@app.post("/api/videos/switch")
async def switch_video(req: VideoSwitchRequest):
    """Updates the global video path and triggers a tracker reset."""
    global VIDEO_PATH, reset_trackers_flag
    new_path = os.path.join("test_videos", req.filename)
    
    if os.path.exists(new_path):
        VIDEO_PATH = new_path
        reset_trackers_flag = True # Tell the video loop to swap out the cv2 object
        return {"status": "success", "video": req.filename}
    
    raise HTTPException(404, detail="Video file not found")

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

@app.get("/video_feed")
async def video_feed(request: Request):
    """Endpoint serving the processed MJPEG video stream."""
    return StreamingResponse(generate_frames(request), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/alerts")
async def get_alerts():
    """Endpoint returning all system alerts in reverse chronological order."""
    return JSONResponse(content={"alerts": list(reversed(detector.alerts))})


# ==========================================
# SECTION 9: AREA MANAGEMENT ENDPOINTS
# ==========================================

class AreaModel(BaseModel):
    id: Optional[str] = None 
    name: Optional[str] = None
    type: str = "polygon" 
    points: Optional[List[List[int]]] = [] 

# --- Restricted Area Routes ---

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

@app.patch("/restricted-areas/{area_id}/toggle")
async def toggle_restricted_area(area_id: str):
    if not is_valid_objectid(area_id): raise HTTPException(400, "Invalid ID")
    area = await areas_collection.find_one({"_id": ObjectId(area_id)})
    if not area: raise HTTPException(404, "Area not found")
    
    new_state = not area.get("is_active", True)
    await areas_collection.update_one({"_id": ObjectId(area_id)}, {"$set": {"is_active": new_state}})
    await update_area_cache()
    return {"status": "ok", "is_active": new_state}

# --- Loitering Area Routes ---

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

@app.patch("/loitering-areas/{area_id}/toggle")
async def toggle_loitering_area(area_id: str):
    if not is_valid_objectid(area_id): raise HTTPException(400, "Invalid ID")
    area = await loitering_collection.find_one({"_id": ObjectId(area_id)})
    if not area: raise HTTPException(404, "Area not found")
    
    new_state = not area.get("is_active", True)
    await loitering_collection.update_one({"_id": ObjectId(area_id)}, {"$set": {"is_active": new_state}})
    await update_area_cache()
    return {"status": "ok", "is_active": new_state}


# ==========================================
# SECTION 10: ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)