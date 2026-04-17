import cv2
import numpy as np
import pickle
import os
import time
import torch
from collections import deque
from typing import List, Dict, Tuple, Optional, Set, Any

# Third-party imports
from ultralytics import YOLO
from cjm_byte_track.core import BYTETracker
# --- NEW IMPORTS FOR FACE RECOGNITION ---
from insightface.app import FaceAnalysis
from milvus_utils import MilvusFaceDB

class RealTimePersonDetector:
    """
    Handles Person Detection (YOLOv8), Tracking (ByteTrack), 
    Identity Association (Simple HSV Histogram Re-ID), and Behavior Analysis.
    Now includes Face Recognition via Milvus.
    """

    def __init__(self, 
                 database_path: str = 'person_database.pkl', 
                 model_path: str = 'yolov8s.pt', 
                 performance_mode: str = 'balanced',
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530"):
        
        print("\n" + "="*50)
        print(f"Initializing Person Detector (Mode: {performance_mode})")
        print("="*50)

        self.performance_mode = performance_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # --- 1. CONFIGURATION SETUP ---
        if self.performance_mode == 'speed':
            self.infer_imgsz = 640
            self.confidence_threshold = 0.45
            self.iou_threshold = 0.60
            self.max_detections = 100
            target_model = 'yolov8n.pt' if model_path == 'yolov8s.pt' else model_path
        elif self.performance_mode == 'accuracy':
            self.infer_imgsz = 832
            self.confidence_threshold = 0.35
            self.iou_threshold = 0.70
            self.max_detections = 300
            target_model = 'yolov8m.pt' if model_path == 'yolov8s.pt' else model_path
        else: # 'performance'
            self.infer_imgsz = 704
            self.confidence_threshold = 0.40
            self.iou_threshold = 0.65
            self.max_detections = 150
            target_model = model_path

        # --- 2. LOAD YOLO MODEL ---
        print(f"[-] Loading YOLO Model: {target_model}")
        if not os.path.exists(target_model):
            if os.path.exists('yolov8n.pt'):
                print(f"    ! Model '{target_model}' not found. Falling back to 'yolov8n.pt'.")
                target_model = 'yolov8n.pt'
            else:
                print(f"    ! Model '{target_model}' not found. Attempting to download/load via Ultralytics.")
        
        self.model = YOLO(target_model)
        self.model.to(self.device)

        # --- 3. TRACKER SETUP ---
        self.tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=90,  
            match_thresh=0.7,
            frame_rate=30
        )
        
        # --- 4. FACE RECOGNITION SETUP (NEW) ---
        
        print("[-] Initializing Face Recognition System...")
        self.face_rec_enabled = False
        self.person_identities = {} # stable_id -> {name, similarity}
        self.rec_interval = 5       # Process faces every 5 frames
        self.min_face_size = 40
        self.face_sim_threshold = 0.60

        try:
            # Initialize InsightFace
            self.face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
            self.face_app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
            
            # Initialize Milvus
            self.milvus_db = MilvusFaceDB(
                host=milvus_host,
                port=milvus_port,
                collection_name="face_embeddings",
                embedding_dim=512
            )
            # Check stats to verify connection
            stats = self.milvus_db.get_collection_stats()
            print(f"    ✓ Connected to Milvus. Faces in DB: {stats['num_entities']}")
            self.face_rec_enabled = True
        except Exception as e:
            print(f"    ✗ Face Recognition Disabled. Error: {e}")
        

        # --- 5. SYSTEM STATE ---
        self.stable_id_features: Dict[int, np.ndarray] = {}  
        self.byte_to_stable_id: Dict[int, int] = {}          
        self.stable_id_last_seen: Dict[int, int] = {}        
        self.next_stable_id = 1
        self.frame_index = 0

        self.active_breaches = set()
        self.alerts = []
        
        # -- Track Status for Colors --
        self.current_breaching_ids = set()
        self.confirmed_loiterers = set()

        # --- 6. BEHAVIOR STATE ---
        self.loitering_state = {}
        
        self.stable_id_positions: Dict[int, deque] = {} 
        self.running_cooldowns: Dict[int, float] = {}   
        self.RUNNING_THRESHOLD = 0.15 
        self.RUNNING_ALERT_COOLDOWN = 0.0 

        # Load legacy database (Optional, kept for compatibility)
        self.known_persons = []
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                self.known_persons = pickle.load(f)

        print("-" * 50)
        print(f"Status: READY ({self.device.upper()})")
        print("=" * 50 + "\n")

    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """Runs YOLOv8."""
        results = self.model(
            frame, 
            conf=self.confidence_threshold, 
            iou=self.iou_threshold, 
            imgsz=self.infer_imgsz, 
            classes=[0], 
            max_det=self.max_detections, 
            augment=False, 
            device=self.device, 
            verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)), 
                            'confidence': float(box.conf)
                        })
        return detections

    def track_persons(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, Dict]:
        """Combines Spatial Tracking, Re-ID, and updates Position History."""
        self.frame_index += 1
        
        if detections:
            det_list = [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']] for d in detections]
            detections_np = np.array(det_list, dtype=float)
        else:
            detections_np = np.empty((0, 5), dtype=float)

        tracks = self.tracker.update(detections_np, frame.shape[:2], frame.shape[:2])
        current_frame_persons = {}
        current_byte_ids = set()

        for track in tracks:
            byte_track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.tlbr)
            current_byte_ids.add(byte_track_id)
            
            bbox = [x1, y1, x2, y2]
            confidence = track.score
            
            appearance_feat = self._extract_appearance_features(frame, bbox)
            stable_id = None
            
            # --- ID ASSOCIATION ---
            if byte_track_id in self.byte_to_stable_id:
                stable_id = self.byte_to_stable_id[byte_track_id]
                if appearance_feat is not None:
                    if stable_id in self.stable_id_features:
                        self.stable_id_features[stable_id] = 0.8 * self.stable_id_features[stable_id] + 0.2 * appearance_feat
                    else:
                        self.stable_id_features[stable_id] = appearance_feat
            else:
                best_match = None
                best_sim = 0.0
                if appearance_feat is not None:
                    current_active_ids = set(self.byte_to_stable_id.values())
                    for candidate_id, prev_feat in self.stable_id_features.items():
                        if candidate_id in current_active_ids: continue
                        frames_lost = self.frame_index - self.stable_id_last_seen.get(candidate_id, 0)
                        if frames_lost >= 200: continue 
                        sim = self._compute_similarity(appearance_feat, prev_feat)
                        is_match = False
                        if frames_lost <= 30 and sim > 0.60: is_match = True
                        elif frames_lost < 200 and sim > 0.75: is_match = True
                        if is_match and sim > best_sim:
                            best_sim = sim
                            best_match = candidate_id

                if best_match is not None:
                    stable_id = best_match
                    self.stable_id_features[stable_id] = 0.7 * self.stable_id_features[stable_id] + 0.3 * appearance_feat
                else:
                    stable_id = self.next_stable_id
                    self.next_stable_id += 1
                    if appearance_feat is not None:
                        self.stable_id_features[stable_id] = appearance_feat
                self.byte_to_stable_id[byte_track_id] = stable_id

            # Update Metadata
            self.stable_id_last_seen[stable_id] = self.frame_index
            
            # --- POSITION HISTORY ---
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if stable_id not in self.stable_id_positions:
                self.stable_id_positions[stable_id] = deque(maxlen=5) 
            self.stable_id_positions[stable_id].append((center_x, center_y))

            current_frame_persons[stable_id] = {
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence
            }

        self._cleanup_memory(current_byte_ids)
        return current_frame_persons

    def recognize_identities(self, frame: np.ndarray, persons: Dict[int, Dict]):
        if not self.face_rec_enabled:
            return

        # Process every N frames to keep FPS high
        if self.frame_index % self.rec_interval != 0:
            return

        for stable_id, pdata in persons.items():
            # If we already identified them in this session with high confidence, skip
            if stable_id in self.person_identities and self.person_identities[stable_id]['similarity'] > 0.85:
                continue

            x1, y1, x2, y2 = pdata['bbox']
            h, w = frame.shape[:2]
            person_roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if person_roi.size == 0: continue

            # 1. Detect faces inside the person's bounding box
            faces = self.face_app.get(person_roi)

            for f in faces:
                # Filter out small/blurry faces
                box = f.bbox
                if (box[2] - box[0]) < self.min_face_size: continue

                embedding = f.embedding
                
                # 2. Search Milvus for a match
                matches = self.milvus_db.search_face(
                    embedding=embedding,
                    top_k=1,
                    threshold=self.face_sim_threshold
                )

                if matches:
                    # MATCH FOUND: Update their identity
                    best_match = matches[0]
                    self.person_identities[stable_id] = {
                        "name": best_match["person_name"],
                        "similarity": best_match["similarity"]
                    }
                    print(f"✨ Match Found: {best_match['person_name']} ({best_match['similarity']:.2f})")
                else:
                    # NO MATCH: Add to Database automatically
                    new_name = f"Person_{stable_id}" # Or handle naming via frontend later
                    self.milvus_db.insert_face(
                        embedding=embedding,
                        stable_id=stable_id,
                        frame_number=self.frame_index,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        person_name=new_name
                    )
                    
                    # Update local state so the UI shows the new ID immediately
                    self.person_identities[stable_id] = {
                        "name": new_name,
                        "similarity": 1.0
                    }
                    print(f"🆕 New Face Registered: {new_name}")

    def detect_running(self, persons: Dict[int, Dict], frame: np.ndarray, camera_name: str):
        """Analyzes movement history to detect running."""
        current_time = time.time()
        
        for pid, pdata in persons.items():
            if pid not in self.stable_id_positions or len(self.stable_id_positions[pid]) < 3:
                continue

            positions = list(self.stable_id_positions[pid])
            dx = positions[-1][0] - positions[0][0]
            dy = positions[-1][1] - positions[0][1]
            dist_pixels = np.sqrt(dx**2 + dy**2)
            
            avg_dist_per_frame = dist_pixels / (len(positions) - 1)
            
            x1, y1, x2, y2 = pdata['bbox']
            box_h = max(1, y2 - y1)
            speed_score = avg_dist_per_frame / box_h
            
            if speed_score > self.RUNNING_THRESHOLD:
                last_alert = self.running_cooldowns.get(pid, 0)
                if current_time - last_alert > self.RUNNING_ALERT_COOLDOWN:
                    self.create_alert_if_new(frame, pid, pdata['bbox'], 
                                         zone_name="General Flow", 
                                         camera_name=camera_name, # Added
                                         alert_type="running")
                    self.running_cooldowns[pid] = current_time
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "RUNNING", (x1, y1 - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    def _cleanup_memory(self, current_byte_ids: Set[int]):
        """Removes old IDs."""
        byte_ids_to_remove = [bid for bid in self.byte_to_stable_id if bid not in current_byte_ids]
        for bid in byte_ids_to_remove: 
            self.byte_to_stable_id.pop(bid)
        
        if self.frame_index % 100 == 0:
            remove_ids = [sid for sid, last in self.stable_id_last_seen.items() 
                          if self.frame_index - last > 300]
            for sid in remove_ids:
                self.stable_id_features.pop(sid, None)
                self.stable_id_last_seen.pop(sid, None)
                
                self.stable_id_positions.pop(sid, None)
                self.running_cooldowns.pop(sid, None)
                self.loitering_state.pop(sid, None)
                # Cleanup identities and states
                self.person_identities.pop(sid, None)
                self.current_breaching_ids.discard(sid)
                self.confirmed_loiterers.discard(sid)

    # --- Feature Extraction Helpers ---
    def _extract_appearance_features(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1: return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
        try:
            roi_resized = cv2.resize(roi, (64, 128))
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            return np.concatenate([hist_h, hist_s, hist_v])
        except Exception:
            return None

    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        if feat1 is None or feat2 is None: return 0.0
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        return max(0.0, np.dot(feat1_norm, feat2_norm))

    # --- Visualization & Utilities ---
    def draw_detections(self, frame: np.ndarray, persons: Dict[int, Dict], 
                    armed_pids: Set[int] = set(), 
                    bag_owner_pids: Set[int] = set(),
                    abandoned_owner_pids: Set[int] = set()) -> np.ndarray: # Add this parameter
        """
        Updated priority-based color system:
        1. Red: Breach, Weapon, OR Abandoned Bag Owner
        2. Orange: Face Recognized
        3. Yellow: Loitering OR Carrying a Bag (Normal)
        4. Green: Default
        """
        for person_id, person_data in persons.items():
            x1, y1, x2, y2 = person_data['bbox']
            
            # --- UPDATED PRIORITY COLOR LOGIC ---
            
            # PRIORITY 1: RED (Breach, Weapon, or Abandoned Bag Owner)
            if (person_id in self.current_breaching_ids or 
                person_id in armed_pids or 
                person_id in abandoned_owner_pids):
                
                color = (0, 0, 255) # Red
                if person_id in abandoned_owner_pids:
                    reason = "ABANDONED OWNER"
                elif person_id in armed_pids:
                    reason = "WEAPON"
                else:
                    reason = "BREACH"
                label_text = f"{reason}! ID {person_id}"

            # PRIORITY 2: ORANGE (Facial Recognition)
            elif person_id in self.person_identities:
                # ... existing orange logic ...
                color = (0, 165, 255)
                label_text = f"ID {person_id}: {self.person_identities[person_id]['name']}"

            # PRIORITY 3: YELLOW (Loitering or Normal Bag Owner)
            elif person_id in self.confirmed_loiterers or person_id in bag_owner_pids:
                color = (0, 255, 255) # Yellow
                reason = "LOITERING" if person_id in self.confirmed_loiterers else "BAG"
                label_text = f"{reason} ID {person_id}"

            # PRIORITY 4: GREEN (Default)
            else:
                color = (0, 255, 0) # Green
                label_text = f"ID {person_id}"

            # --- DRAWING ---
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(text_height, y1 - 5)
            cv2.rectangle(
                frame,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 10, label_y + baseline),
                color,
                -1,
            )
            # Draw Text
            cv2.putText(frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
        return frame

    def detect_breaches_with_ids(self, persons: Dict[int, Dict], areas: List[Dict]) -> List[Tuple]:
        breaches = []
        # Clear breach state for current frame (will be repopulated)
        self.current_breaching_ids.clear()
        
        for area in areas:
            zone_name = area.get("name", "Restricted")
            shape_type = area.get("type", "polygon")
            poly_pts = None
            ellipse_data = None
            if shape_type == "polygon":
                pts = area.get("points", [])
                if len(pts) >= 3: poly_pts = np.array(pts, dtype=np.int32)
            elif shape_type == "ellipse":
                c = area.get("center"); r = area.get("radii")
                if c and r: ellipse_data = (c[0], c[1], r[0], r[1]) 

            if poly_pts is None and ellipse_data is None: continue

            for pid, pdata in persons.items():
                x1, y1, x2, y2 = pdata['bbox']
                feet_x = (x1 + x2) // 2
                feet_y = y2
                is_breach = False
                if shape_type == "polygon":
                    if cv2.pointPolygonTest(poly_pts, (feet_x, feet_y), False) >= 0: is_breach = True     
                elif shape_type == "ellipse":
                    cx, cy, rx, ry = ellipse_data
                    if rx > 0 and ry > 0:
                        if ((feet_x - cx)**2) / (rx**2) + ((feet_y - cy)**2) / (ry**2) <= 1.0: is_breach = True
                
                if is_breach: 
                    breaches.append((pid, zone_name))
                    self.current_breaching_ids.add(pid) # Mark as breaching
                    
        return breaches

    def create_alert_if_new(self, frame: np.ndarray, pid: int, bbox: Tuple, zone_name: str, camera_name: str, alert_type: str = "breach") -> Optional[Dict]:
        key = (pid, zone_name, alert_type) 
        if key in self.active_breaches: return None
        
        if alert_type != "running":
            self.active_breaches.add(key)

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0: return None
        
        _, buf = cv2.imencode('.jpg', crop)
        import base64
        img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        
        alert = {
        'personId': pid,
        'camera': camera_name,  # New field for video filename
        'zone': zone_name,      # Specific zone within that camera
        'type': alert_type,
        'timestamp': time.time(),
        'imageB64': img_b64,
    }
        self.alerts.append(alert)
        return alert
    
    def detect_loitering(self, persons: Dict[int, Dict], areas: List[Dict], frame: np.ndarray, camera_name: str, time_threshold: float = 10.0):
        current_time = time.time()
        current_loiterers = set()
        # We will rebuild confirmed loiterers based on current state to ensure exit = green
        self.confirmed_loiterers.clear()

        for pid, pdata in persons.items():
            feet_x = (pdata['bbox'][0] + pdata['bbox'][2]) // 2
            feet_y = pdata['bbox'][3]
            in_any_zone = False

            for area in areas:
                zone_name = area.get("name", "Loitering Zone")
                shape_type = area.get("type", "polygon")
                is_inside = False
                
                if shape_type == "polygon":
                    pts = area.get("points", [])
                    if len(pts) >= 3:
                        poly_pts = np.array(pts, dtype=np.int32)
                        if cv2.pointPolygonTest(poly_pts, (feet_x, feet_y), False) >= 0: is_inside = True
                elif shape_type == "ellipse":
                    c = area.get("center"); r = area.get("radii")
                    if c and r:
                        cx, cy, rx, ry = c[0], c[1], r[0], r[1]
                        if ((feet_x - cx)**2) / (rx**2) + ((feet_y - cy)**2) / (ry**2) <= 1.0: is_inside = True

                if is_inside:
                    in_any_zone = True
                    current_loiterers.add(pid)
                    if pid not in self.loitering_state:
                        self.loitering_state[pid] = {'zone_name': zone_name, 'start_time': current_time, 'alerted': False}
                    else:
                        state = self.loitering_state[pid]
                        if state['zone_name'] != zone_name:
                            state['zone_name'] = zone_name; state['start_time'] = current_time; state['alerted'] = False
                        
                        duration = current_time - state['start_time']
                        
                        # Check if loitering logic is met for coloring
                        if duration > time_threshold:
                            self.confirmed_loiterers.add(pid)
                        
                        if duration > time_threshold and not state['alerted']:
                            self.create_alert_if_new(frame, pid, pdata['bbox'], zone_name, camera_name, alert_type="loitering") # Added camera_name
                            state['alerted'] = True
                    break 

        for pid in list(self.loitering_state.keys()):
            if pid not in current_loiterers:
                del self.loitering_state[pid]