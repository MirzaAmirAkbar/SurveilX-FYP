import cv2
import numpy as np
import pickle
import os
from ultralytics import YOLO
import time
import torch

class RealTimePersonDetector:
    def __init__(self, database_path='person_database.pkl', model_path='yolov8s.pt', performance_mode='balanced'):
        self.confidence_threshold = 0.4
        self.person_similarity_threshold = 0.7
        self.performance_mode = performance_mode
        
        if self.performance_mode == 'speed':
            self.infer_imgsz = 640
            self.confidence_threshold = 0.45
            self.iou_threshold = 0.6
            self.max_detections = 100
            if model_path == 'yolov8s.pt':
                model_path = 'yolov8n.pt'
        elif self.performance_mode == 'accuracy':
            self.infer_imgsz = 832
            self.confidence_threshold = 0.35
            self.iou_threshold = 0.7
            self.max_detections = 300
            if model_path == 'yolov8s.pt':
                model_path = 'yolov8m.pt'
        else:  # balanced
            self.infer_imgsz = 704
            self.iou_threshold = 0.65
            self.max_detections = 150
        
        print("Loading YOLO model...")
        if not os.path.exists(model_path):
            if os.path.exists('yolov8n.pt'):
                print(f"Model '{model_path}' not found. Falling back to 'yolov8n.pt'.")
                model_path = 'yolov8n.pt'
            else:
                print(f"Model '{model_path}' not found. Proceeding with '{model_path}'.")
        self.model = YOLO(model_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.known_persons = []
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                self.known_persons = pickle.load(f)
            print(f"Loaded {len(self.known_persons)} known persons")
        else:
            print("No person database found.")
        
        self.tracks = {}
        self.next_person_id = 1
        self.active_breaches = set()
        self.alerts = []

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0: return 0.0
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def detect_persons(self, frame):
        results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, imgsz=self.infer_imgsz, classes=[0], max_det=self.max_detections, augment=False, device=self.device, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': float(box.conf)})
        return detections
    
    def track_persons(self, detections, frame):
        current_frame_persons = {}
        unmatched_detections = []
        used_track_ids = set()

        for det in detections:
            best_iou = 0.0
            best_track_id = None
            for track_id, track in self.tracks.items():
                if track_id in used_track_ids: continue
                iou = self._compute_iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= 0.3 and best_track_id is not None:
                self.tracks[best_track_id] = {'bbox': det['bbox'], 'last_seen_ts': time.time()}
                used_track_ids.add(best_track_id)
                current_frame_persons[best_track_id] = {'bbox': det['bbox'], 'confidence': det['confidence']}
            else:
                unmatched_detections.append(det)

        for det in unmatched_detections:
            new_id = self.next_person_id
            self.next_person_id += 1
            self.tracks[new_id] = {'bbox': det['bbox'], 'last_seen_ts': time.time()}
            current_frame_persons[new_id] = {'bbox': det['bbox'], 'confidence': det['confidence']}

        now = time.time()
        stale = [t for t, v in self.tracks.items() if now - v['last_seen_ts'] > 2.5]
        for t in stale: self.tracks.pop(t, None)

        return current_frame_persons

    def enhance_low_light(self, frame, gamma=1.2):
        try:
            if frame.mean() > 90: return frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return np.clip(((img / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
        except: return frame

    def draw_detections(self, frame, persons):
        for person_id, person_data in persons.items():
            x1, y1, x2, y2 = person_data['bbox']
            color = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {person_id}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def draw_area_overlay(self, frame, area, color=(0, 0, 255), alpha=0.3):
        """Draw either a polygon or an ellipse based on area type"""
        overlay = frame.copy()
        name = area.get("name", "Restricted")
        text_pos = None

        shape_type = area.get("type", "polygon")

        if shape_type == "polygon":
            points = area.get("points", [])
            if len(points) < 3: return frame
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)
            
            M = cv2.moments(pts)
            if M["m00"] != 0:
                text_pos = (int(M["m10"] / M["m00"]) - 20, int(M["m01"] / M["m00"]))

        elif shape_type == "ellipse":
            center = area.get("center")
            radii = area.get("radii")
            if center and radii:
                c = (int(center[0]), int(center[1]))
                axes = (int(radii[0]), int(radii[1]))
                cv2.ellipse(overlay, c, axes, 0, 0, 360, color, -1)
                cv2.ellipse(frame, c, axes, 0, 0, 360, color, 2)
                text_pos = (c[0] - 20, c[1])

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        if text_pos:
            cv2.putText(frame, name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame

    def detect_breaches_with_ids(self, persons, areas):
        """Check breaches for both Polygons and Ellipses"""
        breaches = []
        for area in areas:
            zone_name = area.get("name", "Restricted")
            shape_type = area.get("type", "polygon")
            
            # Pre-calculate shape data
            poly_pts = None
            ellipse_data = None
            
            if shape_type == "polygon":
                pts = area.get("points", [])
                if len(pts) >= 3:
                    poly_pts = np.array(pts, dtype=np.int32)
            elif shape_type == "ellipse":
                c = area.get("center")
                r = area.get("radii")
                if c and r:
                    ellipse_data = (c[0], c[1], r[0], r[1]) # cx, cy, rx, ry

            if poly_pts is None and ellipse_data is None:
                continue

            for pid, pdata in persons.items():
                x1, y1, x2, y2 = pdata['bbox']
                # Check feet position
                feet_x = (x1 + x2) // 2
                feet_y = y2
                
                is_breach = False
                
                if shape_type == "polygon":
                    # returns +1 if inside, -1 if outside, 0 on edge
                    if cv2.pointPolygonTest(poly_pts, (feet_x, feet_y), False) >= 0:
                        is_breach = True
                        
                elif shape_type == "ellipse":
                    cx, cy, rx, ry = ellipse_data
                    if rx > 0 and ry > 0:
                        # Equation: (x-cx)^2/rx^2 + (y-cy)^2/ry^2 <= 1
                        val = ((feet_x - cx)**2) / (rx**2) + ((feet_y - cy)**2) / (ry**2)
                        if val <= 1.0:
                            is_breach = True

                if is_breach:
                    breaches.append((pid, zone_name))
                    
        return breaches

    def create_alert_if_new(self, frame, pid, bbox, zone_name):
        key = (pid, zone_name)
        if key in self.active_breaches: return None
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
            'zone': zone_name,
            'timestamp': time.time(),
            'imageB64': img_b64,
        }
        self.alerts.append(alert)
        return alert