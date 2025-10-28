import cv2
import numpy as np
import pickle
import os
from ultralytics import YOLO
import time
import torch

class RealTimePersonDetector:
    def __init__(self, database_path='person_database.pkl', model_path='yolov8s.pt', performance_mode='balanced'):
        """
        Initialize the real-time person detector
        """
        self.confidence_threshold = 0.4
        self.person_similarity_threshold = 0.7
        self.performance_mode = performance_mode  # 'speed' | 'balanced' | 'accuracy'
        # Default per-mode settings
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
        
        # Load YOLO model
        print("Loading YOLO model...")
        if not os.path.exists(model_path):
            if os.path.exists('yolov8n.pt'):
                print(f"Model '{model_path}' not found. Falling back to 'yolov8n.pt'.")
                model_path = 'yolov8n.pt'
            else:
                print(f"Model '{model_path}' not found and fallback 'yolov8n.pt' missing. Proceeding to load '{model_path}' anyway.")
        self.model = YOLO(model_path)

        # Device and precision settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model.to(self.device)
        except Exception:
            pass
        self.use_half = self.device == 'cuda'
        try:
            if hasattr(self.model, 'fuse'):
                self.model.fuse()
        except Exception:
            pass
        try:
            dummy = np.zeros((self.infer_imgsz, self.infer_imgsz, 3), dtype=np.uint8)
            _ = self.model(
                dummy,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.infer_imgsz,
                classes=[0],
                max_det=min(10, self.max_detections),
                augment=False,
                device=self.device,
                verbose=False
            )
        except Exception:
            pass

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass
        
        # Load person database
        self.known_persons = []
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                self.known_persons = pickle.load(f)
            print(f"Loaded {len(self.known_persons)} known persons from database")
        else:
            print("No person database found. Starting fresh.")
        
        self.current_persons = {}
        self.person_id_counter = 0

    def calculate_person_features(self, person_crop):
        """Calculate features for person similarity comparison"""
        if person_crop.size == 0:
            return None, None
            
        person_height, person_width = person_crop.shape[:2]
        aspect_ratio = person_width / person_height if person_height > 0 else 0
        
        hist = cv2.calcHist([person_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_normalized = cv2.normalize(hist, None).flatten()
        
        return aspect_ratio, hist_normalized
    
    def identify_person(self, person_crop):
        """Identify if this person matches any known person"""
        aspect_ratio, hist_normalized = self.calculate_person_features(person_crop)
        
        if aspect_ratio is None:
            return None, 0.0
            
        best_match = None
        best_similarity = 0.0
        
        for known_person in self.known_persons:
            size_similarity = 1 - abs(aspect_ratio - known_person['aspect_ratio']) / max(aspect_ratio, known_person['aspect_ratio'])
            hist_corr = cv2.compareHist(hist_normalized, known_person['histogram'], cv2.HISTCMP_CORREL)
            similarity_score = (size_similarity + hist_corr) / 2
            
            if similarity_score > best_similarity and similarity_score >= self.person_similarity_threshold:
                best_similarity = similarity_score
                best_match = known_person
        
        return best_match, best_similarity
    
    def detect_persons(self, frame):
        """Detect persons in the frame and return detections with IDs"""
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
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0:
                        confidence = float(box.conf)
                        if confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence
                            })
        
        return detections
    
    def track_persons(self, detections, frame):
        """Track persons across frames and assign IDs"""
        current_frame_persons = {}
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            person_crop = frame[y1:y2, x1:x2]
            
            known_person, similarity = self.identify_person(person_crop)
            
            if known_person:
                person_id = known_person['name']
                person_type = "Known"
            else:
                person_id = f"Unknown_{self.person_id_counter}"
                self.person_id_counter += 1
                person_type = "Unknown"
            
            current_frame_persons[person_id] = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'type': person_type,
                'similarity': similarity if known_person else 0.0
            }
        
        return current_frame_persons

    def enhance_low_light(self, frame, gamma=1.2):
        """Enhance low-light frames using CLAHE and gamma correction."""
        try:
            if frame.mean() > 90:
                return frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = np.clip(((img / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)
            return img
        except Exception:
            return frame

    def draw_detections(self, frame, persons):
        """Draw bounding boxes and labels on the frame"""
        for person_id, person_data in persons.items():
            x1, y1, x2, y2 = person_data['bbox']
            confidence = person_data['confidence']
            person_type = person_data['type']
            similarity = person_data['similarity']
            
            if person_type == "Known":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{person_id}"
            if person_type == "Known":
                label += f" ({similarity:.2f})"
            label += f" {confidence:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def draw_translucent_rectangle(self, frame, x, y, width, height, name=None, color=(255, 0, 0), alpha=0.3):
        """Draw a translucent rectangle with an optional name label."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if name:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
            text_x = x + 5
            text_y = y + text_size[1] + 5

            # Optional: draw a background behind the text for readability
            cv2.rectangle(frame, (x, y), (x + text_size[0] + 10, y + text_size[1] + 10), (0, 0, 0), -1)
            cv2.putText(frame, name, (text_x, text_y + 5), font, font_scale, (255, 255, 255), font_thickness)
        return frame


    def check_zone_breach(self, persons, rectangles):
        """Check if any person enters restricted zones."""
        breached_zones = set()
        
        for rect in rectangles:
            rx1, ry1 = rect["x"], rect["y"]
            rx2, ry2 = rx1 + rect["width"], ry1 + rect["height"]
            
            for person_data in persons.values():
                x1, y1, x2, y2 = person_data['bbox']

                # Compute intersection
                inter_x1 = max(x1, rx1)
                inter_y1 = max(y1, ry1)
                inter_x2 = min(x2, rx2)
                inter_y2 = min(y2, ry2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    breached_zones.add(rect.get("name", "Unnamed Zone"))
        
        return breached_zones


    def process_video_source(self, source=0):
        """Process video from various sources"""
        print(f"Starting video processing from source: {source}")

        # Define static rectangles (edit this list anytime)
        self.rectangles = [
            {"x": 484, "y": 204, "width": 137, "height": 138, "name":"area 1"},
            {"x": 8, "y": 185, "width": 105, "height": 168, "name":"area2"},
            # Add more rectangles below as needed:
            # {"x": X, "y": Y, "width": W, "height": H},
        ]


        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                enhanced_frame = self.enhance_low_light(frame)
                detections = self.detect_persons(enhanced_frame)
                persons = self.track_persons(detections, frame)
                frame = self.draw_detections(frame, persons)

                # Check for breaches
                breached_zones = self.check_zone_breach(persons, self.rectangles)
                if breached_zones:
                    breach_text = " | ".join([f"{zone} breached" for zone in breached_zones])
                else:
                    breach_text = "All zones clear"

                # Draw all predefined translucent rectangles with names
                for rect in self.rectangles:
                    frame = self.draw_translucent_rectangle(
                        frame,
                        rect["x"], rect["y"],
                        rect["width"], rect["height"],
                        name=rect.get("name", None)
                    )

                # Info text at top
                info_text = f"Persons: {len(persons)} | Known: {sum(1 for p in persons.values() if p['type'] == 'Known')} | Unknown: {sum(1 for p in persons.values() if p['type'] == 'Unknown')}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Breach text below it
                cv2.putText(frame, breach_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if breached_zones else (0, 255, 0), 2)

                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Real-Time Person Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'captured_frame_{int(time.time())}.jpg', frame)
                    print("Frame saved!")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Video processing ended")

def main():
    print("=== Real-Time Person Detection System ===")
    print("Options:")
    print("1. Webcam (default)")
    print("2. Video file")
    print("3. IP Camera")
    print("4. Exit")
    print("")
    print("Performance modes:  s = speed  |  b = balanced (default)  |  a = accuracy")
    
    choice = input("Enter your choice (1-4): ").strip()
    perf = input("Choose performance mode (s/b/a): ").strip().lower()
    if perf == 's':
        mode = 'speed'
    elif perf == 'a':
        mode = 'accuracy'
    else:
        mode = 'balanced'
    
    detector = RealTimePersonDetector(performance_mode=mode)
    
    if choice == '1' or choice == '':
        print("Starting webcam detection...")
        detector.process_video_source(0)
    elif choice == '2':
        video_path = input("Enter path to video file: ").strip()
        if os.path.exists(video_path):
            detector.process_video_source(video_path)
        else:
            print("Video file not found!")
    elif choice == '3':
        ip_url = input("Enter IP camera URL (e.g., rtsp://192.168.1.100:554/stream): ").strip()
        detector.process_video_source(ip_url)
    elif choice == '4':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
