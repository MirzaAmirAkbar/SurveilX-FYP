import cv2
from weapon_detection import WeaponDetector

# CONFIGURATION
# You will need a .pt file trained on weapons. 
# If you don't have one, search "YOLOv8 weapon detection model" online.
MODEL_PATH = "w1.pt" 
VIDEO_SOURCE = "gun3.mkv"  # Use 0 for webcam, or "path/to/video.mp4"

def main():
    detector = WeaponDetector(model_path=MODEL_PATH, conf_threshold=0.6)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect
        detections = detector.detect_weapons(frame)
        
        # 2. Visualize
        frame = detector.draw_detections(frame, detections)
        
        # 3. Alert Logic (Simple Console Print for testing)
        if detections:
            print(f"⚠️ WEAPON DETECTED: {[d['class'] for d in detections]}")

        cv2.imshow("Weapon Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()