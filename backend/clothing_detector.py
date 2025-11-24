import cv2
import numpy as np

class UnusualClothingDetector:
    def __init__(self):
        # Thresholds for "baggy" or "suspicious" appearance
        # A normal person has a width/height ratio around 0.3 - 0.45
        # Baggy clothes often make a person look wider or "boxier"
        self.ASPECT_RATIO_THRESHOLD = 0.55 
        
    def detect_suspicious_clothing(self, person_crop):
        """
        Analyzes a cropped image of a person to determine if their clothing
        appears unusual (e.g., overly baggy).
        
        Args:
            person_crop (numpy array): Image crop of the detected person.
            
        Returns:
            bool: True if unusual clothing is detected, False otherwise.
            str: Description of the anomaly (e.g., "Baggy Clothing").
        """
        if person_crop.size == 0:
            return False, None

        h, w = person_crop.shape[:2]
        
        # Check 1: Aspect Ratio Analysis
        # Baggy clothes tend to increase the perceived width of a person relative to height.
        aspect_ratio = w / h
        
        if aspect_ratio > self.ASPECT_RATIO_THRESHOLD:
            # Further filter: Ensure it's not just a person bending down
            # (Simple heuristic: check if the crop is reasonably tall enough to be a standing person)
            if h > 100: 
                return True, "Baggy/Bulky Clothing Detected"

        # Check 2: Color Uniformity (Optional Heuristic)
        # Suspicious individuals sometimes wear all-black or very dark monochromatic clothing.
        # We can check the average brightness of the crop.
        gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_crop)
        
        # Threshold for "very dark clothing" (0 is black, 255 is white)
        if avg_brightness < 40: 
             # Combine with width check for higher confidence
             if aspect_ratio > 0.45:
                 return True, "Suspicious Dark/Bulky Attire"

        return False, None

# Example Usage (for testing this file independently):
if __name__ == "__main__":
    # Create a dummy image simulating a wide/baggy person shape
    dummy_person = np.zeros((300, 200, 3), dtype=np.uint8) 
    detector = UnusualClothingDetector()
    is_suspicious, reason = detector.detect_suspicious_clothing(dummy_person)
    print(f"Suspicious: {is_suspicious}, Reason: {reason}")