import cv2
import numpy as np
import time
from ultralytics import YOLO

class WeaponDetector:
    def __init__(self, model_path='yolov8n_weapons.pt', conf_threshold=0.50):
        """
        Initialize the Weapon Detector.
        """
        print(f"[-] Loading Weapon Detector: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.names = self.model.names
            
            # FIX: If the model has weird class names like '0', rename them for display
            if 0 in self.names and self.names[0] == '0':
                self.names[0] = 'Weapon'
                
            print(f"    ✓ Weapon Model Loaded. Classes: {self.names}")
        except Exception as e:
            print(f"    ✗ Error loading weapon model: {e}")
            self.model = None

        self.conf_threshold = conf_threshold

    def detect_weapons(self, frame):
        """
        Runs inference on a frame and returns weapon detections.
        """
        if not self.model:
            return []

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # valid class name from our (potentially renamed) list
                cls_name = self.names[cls_id] 
                conf = float(box.conf[0])
                
                # --- CHANGE IS HERE ---
                # We REMOVED the "if cls_name in [...]" check. 
                # Since this model is ONLY for weapons, we accept everything it finds.
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': cls_name,
                    'confidence': conf
                })
        
        return detections

    def draw_detections(self, frame, detections):
        """
        Visualizes detected weapons on the frame.
        """
        if not detections:
            return frame

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Extract the person ID we mapped in main.py
            pid_str = f" [PID: {det.get('person_id', 'N/A')}]" if 'person_id' in det else ""
            label = f"{det['class'].upper()} ({det['confidence']:.2f}){pid_str}"
            
            # Weapon Alert Color: Deep Purple
            color = (128, 0, 128) 
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw Label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame